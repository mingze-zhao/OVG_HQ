import csv
import os

# 输入的文件夹路径列表（相对路径）
folders = [
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_no_teacher_lr_0.95e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_no_teacher_lr_1.0e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_no_teacher_lr_1.05e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp1_lr0.95e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp1_lr1e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp1_lr1.05e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp2_lr0.95e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp2_lr1e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp2_lr1.05e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp3_lr0.95e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp3_lr1e-4",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/transformer_ttt_pred_query_4_D_D_teacher_temp3_lr1.05e-4",
]  # 请使用文件夹路径列表

# 输出的CSV文件路径
output_file = '/root/autodl-tmp/online-vg/output_gen_segmenttext.csv'

# 创建一个空列表用于保存第二行数据
second_rows = []

# 遍历文件夹路径列表
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)  # 获取文件夹的完整路径
    
    # 确保文件夹存在
    if os.path.isdir(folder_path):
        output_csv_path = os.path.join(folder_path, 'output.csv')  # 文件夹中的 output.csv 文件路径
        
        # 检查 output.csv 文件是否存在
        if os.path.exists(output_csv_path):
            with open(output_csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                # 跳过第一行（如果有标题）
                next(reader)
                next(reader)
                next(reader)
                next(reader)
                next(reader)
                next(reader)
                next(reader)
                next(reader)
                # 获取第二行
                second_row = next(reader, None)
                if second_row:
                    second_rows.append(second_row)
        else:
            print(f"警告: {output_csv_path} 不存在!")

# 将第二行数据写入新的CSV文件
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入第二行数据
    writer.writerows(second_rows)

print(f"所有第二行数据已成功写入到 {output_file}")
