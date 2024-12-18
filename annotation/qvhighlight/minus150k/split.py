import os
import json

def split_jsonl_file(input_path, output_dir, num_splits=8):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有数据
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算每一份的大小
    total_lines = len(lines)
    split_size = (total_lines + num_splits - 1) // num_splits  # 向上取整

    # 将数据均分并写入新的文件
    for i in range(num_splits):
        subset_lines = lines[i * split_size : (i + 1) * split_size]
        output_path = os.path.join(output_dir, f"subset{i}.jsonl")

        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(subset_lines)

        print(f"已写入 {len(subset_lines)} 行到 {output_path}")

# 使用示例
input_jsonl = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/minus150k_videos.jsonl"  # 输入文件路径
output_folder = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/minus150k/"  # 输出文件夹路径

split_jsonl_file(input_jsonl, output_folder, num_splits=8)
