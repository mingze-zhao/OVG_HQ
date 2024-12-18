import json

# 输入文件路径


import json

def merge_old_query(current_file, refinement_file):
    """
    将 refinement_file 中的 old_query 根据 qid 追加到 current_file 的每个样本里，并输出到 output_file。

    参数:
    current_file (str): 当前 jsonl 文件路径。
    refinement_file (str): 包含 old_query 的 jsonl 文件路径。
    output_file (str): 输出文件路径。
    """
    # 读取 refinement 文件，构建一个 qid 到 old_query 的映射
    output_file = current_file[:-6]+"_old_query.jsonl"
    qid_to_old_query = {}
    with open(refinement_file, 'r', encoding='utf-8') as f_ref:
        for line in f_ref:
            sample = json.loads(line)
            qid = sample.get("qid")
            old_query = sample.get("old_query")
            if qid and old_query:
                qid_to_old_query[qid] = old_query

    # 读取 current 文件，并根据 qid 追加 old_query
    with open(current_file, 'r', encoding='utf-8') as f_current, open(output_file, 'w', encoding='utf-8') as f_output:
        for line in f_current:
            sample = json.loads(line)
            qid = sample.get("qid")
            if qid in qid_to_old_query:
                sample["old_query"] = qid_to_old_query[qid]
            # 写入新的jsonl文件
            f_output.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"已成功处理文件: {current_file} 并输出到 {output_file}")

current_file_list = ['/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand1.jsonl', '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/train_rand4.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand1.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/val_rand4.jsonl',
'/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cartoon_rand1.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cartoon_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cartoon_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cartoon_rand4.jsonl',
'/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cinematic_rand1.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cinematic_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cinematic_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_cinematic_rand4.jsonl',
'/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_realistic_rand1.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_realistic_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_realistic_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_realistic_rand4.jsonl',
'/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_scribble_rand1.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_scribble_rand2.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_scribble_rand3.jsonl','/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/test_scribble_rand4.jsonl']
refinement_file = '/mnt/data/jiaqi/online-vg/annotation/qvhighlight_icq/output_with_refinement_final.jsonl'

for i in current_file_list:
    merge_old_query(i, refinement_file)