# import json
# import random

# # 定义输入文件路径和输出文件路径
# input_file = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/output_with_refinement_final.jsonl'  # 输入文件路径，需根据你的文件路径调整
# train_file = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand4.jsonl'
# val_file = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand4.jsonl'
# test_file = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_rand4.jsonl'

# # 读取 JSONL 文件
# data_samples = []
# with open(input_file, 'r', encoding='utf-8') as file:
#     for line in file:
#         data_samples.append(json.loads(line.strip()))

# # 随机打乱数据顺序
# random.shuffle(data_samples)

# # 计算数据集的拆分比例
# total_samples = len(data_samples)
# train_size = int(0.7 * total_samples)
# val_size = 219
# test_size = 219

# # 拆分数据集
# train_data = data_samples[:train_size]
# val_data = data_samples[train_size:train_size + val_size]
# test_data = data_samples[train_size + val_size:]

# # 将数据集保存为 JSONL 格式文件
# def save_jsonl(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         for item in data:
#             file.write(json.dumps(item) + '\n')

# # 保存训练集、验证集和测试集
# save_jsonl(train_data, train_file)
# save_jsonl(val_data, val_file)
# save_jsonl(test_data, test_file)

# print(f"数据集已拆分并保存为:\n训练集: {train_file}\n验证集: {val_file}\n测试集: {test_file}")

import json
import random
from pathlib import Path

# 定义文件路径
input_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/output_with_refinement_final1.jsonl'
train_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand4.jsonl'
val_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand4.jsonl'
test_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_rand4.jsonl'
test_cinematic_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cinematic_rand4.jsonl'
test_cartoon_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cartoon_rand4.jsonl'
test_realistic_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_realistic_rand4.jsonl'
test_scribble_file_path = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_scribble_rand4.jsonl'

# 读取所有jsonl文件行
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 将数据写入jsonl文件
def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# 读取数据集
data = read_jsonl(input_file_path)

# 随机打乱数据集
random.shuffle(data)

# 计算训练集、验证集、测试集的划分数量
total_samples = len(data)
train_size = int(total_samples * 0.7)
val_size = 218
test_size = total_samples - train_size - val_size

# 划分数据集
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# 保存初始的训练集、验证集和测试集
write_jsonl(train_data, train_file_path)
write_jsonl(val_data, val_file_path)
write_jsonl(test_data, test_file_path)

# 将训练集和验证集的样本拆分为四种风格
def split_styles(data, qid_key='qid'):
    new_data = []
    for sample in data:
        qid = sample[qid_key]
        for style_type in ['cinematic', 'cartoon', 'realistic', 'scribble']:
            # 生成新样本
            new_sample = sample.copy()
            img_dir = f"qid{qid}_{style_type}.npz"
            # new_sample['img_dir'] = img_dir
            new_sample['qid'] = f"{qid}_{style_type}"
            # 删除原有风格路径
            del new_sample['cinematic']
            del new_sample['cartoon']
            del new_sample['realistic']
            del new_sample['scribble']
            new_data.append(new_sample)
    return new_data

# 拆分训练集和验证集并保存
train_data_split = split_styles(train_data)
val_data_split = split_styles(val_data)

write_jsonl(train_data_split, train_file_path)
write_jsonl(val_data_split, val_file_path)

# 将测试集拆分为四个不同风格的文件
def split_test_set(data, qid_key='qid'):
    test_splits = {
        'cinematic': [],
        'cartoon': [],
        'realistic': [],
        'scribble': []
    }
    for sample in data:
        qid = sample[qid_key]
        for style_type in test_splits.keys():
            new_sample = sample.copy()
            img_dir = f"qid{qid}_{style_type}.npz"
            # new_sample['img_dir'] = img_dir
            new_sample['qid'] = f"{qid}_{style_type}"
            # 删除其他风格路径
            for key in test_splits.keys():
                del new_sample[key]
            # 将样本添加到对应风格列表中
            test_splits[style_type].append(new_sample)
    return test_splits

# 拆分测试集并分别保存
test_data_split = split_test_set(test_data)
write_jsonl(test_data_split['cinematic'], test_cinematic_file_path)
write_jsonl(test_data_split['cartoon'], test_cartoon_file_path)
write_jsonl(test_data_split['realistic'], test_realistic_file_path)
write_jsonl(test_data_split['scribble'], test_scribble_file_path)

# 提示用户操作完成
print("数据集已成功划分并拆分为四种风格，生成的文件如下：")
print(f"训练集: {train_file_path}")
print(f"验证集: {val_file_path}")
print(f"测试集: {test_file_path}")
print(f"测试集风格文件: {test_cinematic_file_path}, {test_cartoon_file_path}, {test_realistic_file_path}, {test_scribble_file_path}")
