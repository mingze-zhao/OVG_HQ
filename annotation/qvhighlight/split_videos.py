import json
import math

def read_jsonl(file_path):
    """读取 JSONL 文件并返回所有行数据的列表"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_data(data, n_splits):
    """将数据均分成 n_splits 份"""
    split_size = math.ceil(len(data) / n_splits)  # 每份的大小
    return [data[i:i + split_size] for i in range(0, len(data), split_size)]

def write_jsonl(file_path, data):
    """将数据写入 JSONL 文件"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_empty_jsonl(file_path):
    """生成一个空的 JSONL 文件"""
    with open(file_path, 'w') as f:
        f.write('')  # 写入空文件

def split_jsonl_file(input_file, output_prefix, n_splits=8):
    """将输入的 JSONL 文件均分成 n_splits 份，并生成空 JSONL 文件"""
    # 读取原始 JSONL 文件
    data = read_jsonl(input_file)

    # 将数据均分
    split_data_list = split_data(data, n_splits)

    # 生成 n_splits 个 JSONL 文件，并将数据写入
    for i, split in enumerate(split_data_list):
        output_file = f"{output_prefix}_part_{i + 1}.jsonl"
        write_jsonl(output_file, split)

    # 再生成 n_splits 个空的 JSONL 文件
    for i in range(n_splits):
        empty_file = f"{output_prefix}_empty_{i + 1}.jsonl"
        create_empty_jsonl(empty_file)

    print(f"已生成 {n_splits} 个包含数据的 JSONL 文件和 {n_splits} 个空的 JSONL 文件。")

# 示例用法
input_file = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/convert5.jsonl"  # 替换为你的输入文件路径
output_prefix = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/subset5/output"  # 输出文件的前缀名

# 调用函数，均分为 8 份并生成新文件和空文件
split_jsonl_file(input_file, output_prefix, n_splits=8)
