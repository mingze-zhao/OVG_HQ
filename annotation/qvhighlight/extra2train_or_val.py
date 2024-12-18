import os
import json
import shutil

# 定义路径
video_dir = "/mnt/data/jiaqi/online-vg/data/QVHighlight/generate_videos/extra"
jsonl_file = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/highlight_val_release.jsonl"
target_dir = "/mnt/data/jiaqi/online-vg/data/QVHighlight/generate_videos/valset"

# 读取 JSONL 文件，获取所有的 qid
with open(jsonl_file, 'r') as f:
    qids = {json.loads(line)['qid'] for line in f}

# 遍历视频文件，检查是否在 qid 列表中
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        # 提取 qid
        qid = int(video_file.split('.')[0][3:])  # 假设格式为 qid123.mp4
        
        # 检查 qid 是否在 JSONL 数据中
        if qid in qids:
            # 定义目标文件路径
            target_file = os.path.join(target_dir, video_file)
            
            # 复制文件到目标目录（覆盖已有文件）
            shutil.copy2(os.path.join(video_dir, video_file), target_file)
            print(f"Copied {video_file} to {target_dir}")

print("Done!")
