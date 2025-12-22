

# import open3d as o3d

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector([[0,0,0], [1,0,0], [0,1,0]])
# o3d.visualization.draw_geometries([pcd]) 


# from openai import OpenAI
# client = OpenAI(
#     api_key="这里是能用AI的api_key",
#     base_url="https://ai.nengyongai.cn/v1"
# )

# response = client.chat.completions.create(
#     messages=[
#     	# 把用户提示词传进来content
#         {'role': 'user', 'content': "鲁迅为什么打周树人？"},
#     ],
#     model='gpt-4',  # 上面写了可以调用的模型
#     stream=True  # 一定要设置True
# )

# for chunk in response:
#     print(chunk.choices[0].delta.content, end="", flush=True)
import gzip 
import pickle
detections_path = '/home/zhengwu/Desktop/concept-graphs/Datasets/Replica/room0/gsa_detections_ram_mobilesam_withbg_allclasses/frame000000.pkl.gz'
with gzip.open(detections_path, "rb") as f:
    gobs = pickle.load(f)
print(gobs)
print('finish')
