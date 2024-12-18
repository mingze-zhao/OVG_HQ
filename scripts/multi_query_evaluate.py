import os
eval_split_name = "val"

# 定义模型路径列表
model_paths = [
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/lstm_lstm/checkpoint/epoch17.ckpt",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/sa_sa/checkpoint/epoch8.ckpt",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/ttt_mlp/checkpoint/epoch16.ckpt",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/sa_mlp/checkpoint/epoch9.ckpt",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/lstm_mlp/checkpoint/epoch21.ckpt",
]
results_dir = [
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/lstm_lstm",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/sa_sa",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/ttt_mlp",
    # "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/sa_mlp",
    "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/lstm_mlp",
]

# 定义查询类型列表
eval_query_types = ["ori_image", "gen_image", "ori_segment", "gen_segment"]

# 直接copy 实验路径下的config文件路径
config_path = "/root/autodl-tmp/online-vg/exp/Last/add_post_real_mr_ttt/qvhightlight_unify_train_all/3407/lstm_lstm/clip_slowfast_online_vg_qvhighlight_unify.yml"

# eval_path 保持和config_path里面的eval_path一致
eval_path = "/root/autodl-tmp/online-vg/annotation/qvhighlight_segment/highlight_segment_val_release_rand2.jsonl"

# 循环执行命令
for idx, model_path in enumerate(model_paths):
    for eval_query_type in eval_query_types:
        command = f"python training/evaluate.py --config {config_path} " \
                    f"--model_path {model_path} " \
                    f"--eval_split_name {eval_split_name} " \
                    f"--eval_path {eval_path} " \
                    f"--results_dir {results_dir[idx]} " \
                  f"--eval_query_type {eval_query_type}" 
    print(f"Executing: {command}")
    os.system(command)  # 执行命令