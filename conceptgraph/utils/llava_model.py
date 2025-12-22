# try:
#     LLAVA_PYTHON_PATH = os.environ["LLAVA_PYTHON_PATH"]
# except KeyError:
#     print("Please set the environment variable LLAVA_PYTHON_PATH to the path of the LLaVA repository")
#     sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import torchvision
from torch import nn 
import warnings
# 过滤特定的警告信息
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
import re
import sys
sys.path.append('/home/zhengwu/Desktop/concept-graphs/LLaVA')

from conceptgraph.llava.llava.utils import disable_torch_init
from conceptgraph.llava.llava.model.builder import load_pretrained_model
from conceptgraph.llava.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from submodules.llava.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conceptgraph.llava.llava.conversation import conv_templates

class LLaVaChat():
    """
    LLaVa (Large Language-and-Vision Assistant) 聊天模型封装类。
    用于处理多模态查询，结合图像内容和文本提示生成自然语言描述。
    """
    # Model Constants - 定义特殊 Token
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    def __init__(self, model_path):
        """
        初始化 LLaVaChat 模型
        :param model_path: 模型权重的本地路径
        """
        disable_torch_init() # 禁用某些 Torch 初始化以加速加载

        self.model_name = get_model_name_from_path(model_path)  
        # 加载预训练模型、分词器和图像处理器
        # load_4bit=True: 使用 4-bit 量化加载模型，显著降低显存占用
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, device="cuda", load_4bit=True)

        # 根据模型名称自动选择对应的对话模板 (Conversation Template)
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
        """
        预处理图像以适配 LLaVa 模型输入
        """
        x = process_images(
            images,
            self.image_processor,
            self.model.config)

        return x.to(self.model.device, dtype=torch.float16)

    def __call__(self, query, image_features, image_sizes):
        """
        前向推理函数：调用模型生成回答
        :param query: 文本提示 (Prompt)
        :param image_features: 预处理后的图像特征
        :param image_sizes: 原始图像尺寸列表
        :return: 模型生成的文本回答
        """
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.

        conv = conv_templates[self.conv_mode].copy()

        # 构建包含特殊 Token 的输入提示
        if self.model.config.mm_use_im_start_end:
            inp = LLaVaChat.DEFAULT_IM_START_TOKEN +\
                  LLaVaChat.DEFAULT_IMAGE_TOKEN +\
                  LLaVaChat.DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = LLaVaChat.DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 将提示文本转换为 token IDs
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, LLaVaChat.IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.temperature = 0 # 温度设为 0 以获得确定性输出
        self.max_new_tokens = 512 # 最大生成长度
        with torch.inference_mode():
            # 生成回答
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
if __name__ == '__main__': 
    mode == "captions"
    llava_chat = LLaVaChat(model_path="/home/zhengwu/Desktop/concept-graphs/llava_v1.6_vicuna") 
    image_path = '/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/results/frame000000.jpg'
    image = Image.open(image_path).convert('RGB')
    ### caption
    image_sizes = [image.size]
    image_tensor = llava_chat.preprocess_image([image]).to("cuda", dtype=torch.float16)
    template = {}

    if mode == "category":
        # 模式1: 类别发现
        # 询问模型图像中有哪些主要对象类别，用于后续 GroundingDINO 的检测
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
        text = llava_chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        # 清理输出文本
        template["categories"] = re.sub(r'\s+', ' ', text.replace("<s>", "").replace("</s>", "").replace("-", "").strip())
    elif mode == "captions":
        # 模式2: 详细描述
        # 对单个对象进行外观、材质等详细描述
        query = '''Please describe the {class_name} in the image as thoroughly as possible:Include color, shape, material, and approximate size.
            Surface texture, visible components and structure, possible purpose/function, current state (e.g., open/closed, full/empty, damaged/intact).
            Relationship to nearby objects or surfaces (e.g., on a table, against a wall, hanging from a hook).
            Output in concise, natural language without unnecessary prefixes or suffixes.'''
        # query = query_base + "\n" + query_tail
        text = llava_chat(query=query.format(class_name='chair'), image_features=image_tensor, image_sizes=image_sizes)
        # template["id"] = mask_id
        template["description"] = text.replace("<s>", "").replace("</s>", "").strip() 
        print('finish')
