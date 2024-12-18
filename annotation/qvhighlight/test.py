import json

def read_qids_from_jsonl(filepath):
    """
    从 jsonl 文件中读取 qid 列表
    """
    qids = set()  # 使用集合来保存 qid 以提高查找效率
    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'qid' in data:
                qids.add(data['qid'])
    return qids

def find_unique_qids(file1, file2):
    """
    找到两个文件中不共同存在的 qid
    """
    qids_file1 = read_qids_from_jsonl(file1)
    qids_file2 = read_qids_from_jsonl(file2)
    
    # 找到不共同存在的 qid
    unique_qids_file1 = qids_file1 - qids_file2  # 只在 file1 中的 qid
    unique_qids_file2 = qids_file2 - qids_file1  # 只在 file2 中的 qid
    
    return unique_qids_file1, unique_qids_file2

def main(file1, file2, output_file):
    unique_qids_file1, unique_qids_file2 = find_unique_qids(file1, file2)
    
    # 输出结果
    with open(output_file, 'w') as out_file:
        out_file.write("不共同存在的 qid（只在 {} 中存在）:\n".format(file1))
        for qid in unique_qids_file1:
            out_file.write(f"qid: {qid}\n")
        
        out_file.write("\n不共同存在的 qid（只在 {} 中存在）:\n".format(file2))
        for qid in unique_qids_file2:
            out_file.write(f"qid: {qid}\n")
    
    print(f"结果已写入到 {output_file}")

if __name__ == "__main__":
    # 输入两个 jsonl 文件路径
    file1 = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/convert1.jsonl"
    file2 = "/mnt/data/jiaqi/online-vg/annotation/qvhighlight/test.jsonl"
    output_file = "unique_qids.txt"  # 输出结果文件

    main(file1, file2, output_file)
