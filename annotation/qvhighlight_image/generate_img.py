import os
import json
import torch
from diffusers import StableDiffusion3Pipeline
# 指定设备
device = "cuda"

# 加载 Stable Diffusion 模型
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
)
pipe = pipe.to(device)

# 设置基础路径
base_path = "/mnt/data/jiaqi/online-vg/data/QvHighlight_Image/generated_image"  # 替换为实际路径

# 创建四种风格的目录
styles = ["scribble", "cartoon", "cinematic", "realistic"]
style_prompts = {
    "scribble": "A simple scribble drawing of",
    "cartoon": "A cartoon style illustration of",
    "cinematic": "A cinematic shot of",
    "realistic": "A highly realistic image of",
}

for style in styles:
    os.makedirs(os.path.join(base_path, f"val_style_{style}"), exist_ok=True)

# 读取 JSONL 文件
jsonl_file = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight_image/split_jsonl_for_generate_part_4.jsonl"  # 替换为你的 JSONL 文件路径

# 遍历每一行样本
with open(jsonl_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        qid = data["qid"]
        query = data["query"]
        
        # 生成四种风格的图片
        for style, prompt_prefix in style_prompts.items():
            prompt = f"{prompt_prefix} {query}"
            image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
            
            # 保存图片
            output_dir = os.path.join(base_path, f"val_style_{style}")
            image.save(os.path.join(output_dir, f"qid{qid}.jpg"))

print("图片生成完成并保存到指定路径。")
