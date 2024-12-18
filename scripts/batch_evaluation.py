import os
import subprocess
import yaml
import re

# 输入路径
base_path = '/mnt/data/jiaqi/online-vg/exp/Last/add_post_real_mr_hd_ttt/qvhightlight_segment_text/transformer_ttt_4_D_pred'  # 请替换为您的路径

# 指标顺序
header_order = [
    "Epoch",  # 添加 Epoch 列
    "OFF3-R1@0.5", "On3-R1@0.5-pos", "On3-R1@0.5-pos_gamma", "On3-R1@0.5-zero", "On3-R1@0.5-zero_gamma",
    "mAP@0.5", "On3-mAP@0.5-pos", "On3-mAP@0.5-pos_gamma", "On3-mAP@0.5-zero", "On3-mAP@0.5-zero_gamma",
    "HL-min-VeryGood-mAP", "HL-min-VeryGood-Hit1",
]

# 遍历 base_path 下的所有一级子文件夹
# for subdir in os.listdir(base_path):
    # sub_path = os.path.join(base_path, subdir)
sub_path = base_path
# if not os.path.isdir(sub_path):
#     continue  # 跳过非文件夹项

# 查找config文件路径
yml_files = [f for f in os.listdir(sub_path) if f.endswith('.yml')]
# if not yml_files:
#     print(f"No yml file found in {sub_path}. Skipping this folder.")
#     continue
yml_path = os.path.join(sub_path, yml_files[0])

# 从config文件读取eval_path
with open(yml_path, 'r') as file:
    config = yaml.safe_load(file)
eval_path = config.get('eval_path')
# if not eval_path:
#     print(f"No eval_path found in the config file at {yml_path}. Skipping this folder.")
#     continue

# 获取所有以"epoch"开头的checkpoint文件并排序
ckpt_dir = os.path.join(sub_path, 'checkpoint')
# if not os.path.exists(ckpt_dir):
#     print(f"No checkpoint directory found in {sub_path}. Skipping this folder.")
#     continue

# epoch_files = sorted(
#     [f for f in os.listdir(ckpt_dir) if re.match(r'^best.*', f)],
#     key=lambda x: int(re.search(r'\d+', x).group())
# )
epoch_files = [f for f in os.listdir(ckpt_dir) if re.match(r'^best.*', f)]

# 执行实验
for epoch_file in epoch_files:
    ckpt_path = os.path.join(ckpt_dir, epoch_file)
    command = [
        'python', 'training/evaluate.py',
        '--config', yml_path,
        '--model_path', ckpt_path,
        '--eval_split_name', 'val',
        '--eval_path', eval_path
    ]
    
    print(f"Executing: {' '.join(command)}")
    subprocess.run(command)
