import os
import re
import sys
import json
import random
import argparse
import cv2
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image  
from dataclasses import dataclass, field
from typing import Tuple, Type
import open3d as o3d
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import torchvision
from torch import nn
from loguru import logger
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from submodules.segment_anything.sam2.build_sam import build_sam2
from submodules.segment_anything.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from submodules.segment_anything.sam2.sam2_image_predictor import SAM2ImagePredictor
from submodules.groundingdino.groundingdino.util.inference import Model
from submodules.llava.llava.utils import disable_torch_init
from submodules.llava.llava.model.builder import load_pretrained_model
from submodules.llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from submodules.llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from submodules.llava.llava.conversation import conv_templates

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "./submodules/open_clip/open_clip_pytorch_model.bin"
    clip_n_dims: int = 512
   
class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type, 
            self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(args.device)
        self.clip_n_dims = self.config.clip_n_dims

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_texts(self, class_ids, classes):
        with torch.no_grad():
            tokenized_texts = torch.cat([self.tokenizer(classes[class_id]) for class_id in class_ids]).to(args.device)
            text_feats = self.model.encode_text(tokenized_texts)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return text_feats

class LLaVaChat():
    # Model Constants
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    def __init__(self, model_path):
        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)  
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, device="cuda")

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def preprocess_image(self, images):
        x = process_images(
            images,
            self.image_processor,
            self.model.config)

        return x.to(self.model.device, dtype=torch.float16)

    def __call__(self, query, image_features, image_sizes):
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.

        conv = conv_templates[self.conv_mode].copy()

        if self.model.config.mm_use_im_start_end:
            inp = LLaVaChat.DEFAULT_IM_START_TOKEN +\
                  LLaVaChat.DEFAULT_IMAGE_TOKEN +\
                  LLaVaChat.DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = LLaVaChat.DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, LLaVaChat.IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.temperature = 0
        self.max_new_tokens = 512
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_features,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        return outputs

def describe_LLAVA(mask_id, image, chat:LLaVaChat, class_i, class_j, Cord_i, Cord_j, mode):

    ### caption
    image_sizes = [image.size]
    image_tensor = chat.preprocess_image([image]).to("cuda", dtype=torch.float16)
    template = {}

    if mode == "category":
        query_base = """Identify and list only the main object categories clearly visible in the image."""

        query_tail = """
        Provide only the category names, separated by commas.
        Only list the main object categories in the image.
        Maximum 10 categories, focus on clear, foreground objects
        Each category should be listed only once, even if multiple instances of the same category are present.
        Avoid overly specific or recursive descriptions.
        Do not include descriptions, explanations, or duplicates.
        Do not include quotes, brackets, or any additional formatting in the output.
        Examples:
        Chair, Table, Window
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        template["categories"] = re.sub(r'\s+', ' ', text.replace("<s>", "").replace("</s>", "").replace("-", "").strip())

    if mode == "captions":
        query_base = """Describe the visible object in front of you, 
        focusing on its spatial dimensions, visual attributes, and material properties."""
        
        query_tail = """
        The object is typically found in indoor scenes and its category is {class_i}.
        Briefly describe the object within ten word. Keep the description concise.
        Focus on the object's appearance, geometry, and material. Do not describe the background or unrelated details.
        Ensure the description is specific and avoids vague terms.
        Examples: 
        a closed wooden door with a glass panel;
        a pillow with a floral pattern;
        a wooden table;
        a gray wall.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query.format(class_i=class_i), image_features=image_tensor, image_sizes=image_sizes)
        template["id"] = mask_id
        template["description"] = text.replace("<s>", "").replace("</s>", "").strip()

    elif mode == "relationships":
        query_base = """There are two objects with category and 2D coordinate, 
        paying close attention to the positional relationship between two selected objects."""
        query_tail = """
        You are capable of analyzing spatial relationships between objects in an image.

        In the given image, there are two boxed objects:
        - The object selected by the red box is [{class_i}], and its bounding box coordinates are {bbox1}.
        - The object selected by the blue box is [{class_j}], and its bounding box coordinates are {bbox2}.

        Note: The bounding box coordinates are in the format (x_min, y_min, x_max, y_max), where (x_min, y_min) represents the top-left corner of the box and (x_max, y_max) represents the bottom-right corner of the box.

        The spatial relationship between [{class_i}] and [{class_j}] may include, but is not limited to, the following types:
        - "Above" means Object A is located higher in vertical position (y_min smaller).
        - "Below" means Object A is located lower in vertical position (y_min larger).
        - "Left" means Object A's x_min is smaller than Object B's x_min.
        - "Right" means Object A's x_min is larger than Object B's x_min.
        - "Inside" means Object A's bounding box is fully contained within Object B's bounding box.
        - "Contains" means Object A's bounding box fully contains Object B's bounding box.
        - "Next to" means the distance between boxes is very small, without overlap.

        Please provide the output in the following format:
        Coarse: The spatial relationship between {class_i} and {class_j}; Fine: A detailed description of the relationship (optional).

        Example output:
        Coarse: The cup is on the table; Fine: The cup is resting near the center of the table, with its handle facing outward.
        Coarse: The book is under the lamp; Fine: The book lies directly beneath the lamp, slightly tilted, as if recently placed.
        Coarse: The cat is next to the sofa; Fine: The cat is sitting closely beside the sofa's left armrest, partially leaning on it.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query.format(class_i=class_i, class_j=class_j, bbox1=Cord_i, bbox2=Cord_j), image_features=image_tensor, image_sizes=image_sizes)
        template["id_pair"] = mask_id
        template["relationship"] = text.replace("<s>", "").replace("</s>", "").strip()

    return template



