import csv

# 输入的CSV文件路径列表
dir_name = "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp3_lr0.95e-4/"
csv_files = ['metricstext.csv', 'metrics_ori_image.csv', 'metrics_ori_segment.csv', 'metrics_gen_image.csv', 'metrics_gen_segment.csv','metrics_ori_imagetext.csv', 'metrics_ori_segmenttext.csv', 'metrics_gen_imagetext.csv', 'metrics_gen_segmenttext.csv']  # 在这里替换成你的文件路径
new_csv_files = []
for csv_file in csv_files:
    csv_file = dir_name + csv_file
    new_csv_files.append(csv_file)
csv_files = new_csv_files 

# 输出的CSV文件路径
output_file = dir_name + 'output.csv'

# 创建一个空列表用于保存第二行数据
second_rows = []

# 读取每个CSV文件并提取第二行
for file in csv_files:
    with open(file, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 跳过第一行（如果有标题）
        next(reader)
        # 获取第二行
        second_row = next(reader, None)
        if second_row:
            second_rows.append(second_row)

# 将第二行数据写入新的CSV文件
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入第二行数据
    writer.writerows(second_rows)

print(f"第二行数据已成功写入到 {output_file}")