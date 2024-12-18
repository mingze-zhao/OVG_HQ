import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
#读取jsonl文件
import json
import os
from tqdm import tqdm

input_jsonl = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/minus150k/subset1.jsonl"
output_video_path = "/mnt/data/jiaqi/online-vg/data/QVHighlight/generate_videos/minus150k"
# prompt = "In a room decorated for a baby boy celebration, a Muslim woman is organizing clothes into bags. The scene is filled with baby-themed decorations, including blue and silver balloons with phrases like 'Baby Boy' and star-shaped designs. Gift bags are placed around the room, one with floral designs and another clearly marked with 'Baby Boy' on it. The woman works quietly in the background, sorting through items, likely preparing or tidying up after the event. The bright daylight filtering in through the windows adds a sense of warmth and joy, as the room exudes celebration and care for the new arrival."
device = "cuda"
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16,
    cache_dir='/mnt/data/jiaqi/cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V',
    local_files_only=True,
)
pipe.to(device)

# pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()



# export_to_video(video, "outputi2v_4.mp4", fps=8)

with open(input_jsonl, "r") as f:
    data = [json.loads(line) for line in f]
    for i in tqdm(range(len(data))):
        prompt = data[i]["query"]
        qid = data[i]["qid"]
        video_path = output_video_path + "/" + "qid" + str(qid) + ".mp4"
        if os.path.exists(video_path):
            continue
        print("qid" + str(qid) + ".mp4")
        image = load_image(image=os.path.join("/mnt/data/jiaqi/online-vg/data/QVHighlight/frame",f"qid{qid}.jpg"))
        video = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        export_to_video(video, video_path, fps=8)