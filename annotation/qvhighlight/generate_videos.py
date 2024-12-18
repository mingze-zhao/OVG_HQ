import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import os
import time
#读取jsonl文件
import json
input_jsonl = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/convert_extra.jsonl"
output_video_path = "/mnt/data/jiaqi/online-vg/data/QVHighlight/generate_videos/extra"

# delay_hours = 2  # 延时时间（小时）
# delay_seconds = delay_hours * 60 * 60  # 转换为秒

# print(f"程序将在 {delay_hours} 小时后启动...")
# time.sleep(delay_seconds)  # 等待6\2小时

device = "cuda"
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
    cache_dir='/mnt/data/jiaqi/cache/huggingface/hub',
    local_files_only=True,
    # force_download=True,
#      mirror="tuna"
)
pipe.to(device)

pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
with open(input_jsonl, "r") as f:
    data = [json.loads(line) for line in f]
    for i in range(len(data)):
        prompt = data[i]["query"]
        qid = data[i]["qid"]
        video_path = output_video_path + "/" + "qid" + str(qid) + ".mp4"
        if os.path.exists(video_path):
            continue
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        export_to_video(video, video_path, fps=8)

