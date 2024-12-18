
# import json
# import matplotlib.pyplot as plt
# import numpy as np

# def read_jsonl(file_path):
#     relevant_windows_lengths = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             # 读取 relevant_windows 并计算每个窗口的长度
#             relevant_windows = data.get("relevant_windows", [])
#             for window in relevant_windows:
#                 length = window[1] - window[0]  # 计算窗口长度
#                 relevant_windows_lengths.append(length)
#     return relevant_windows_lengths

# def plot_histogram_binned(data, bin_size=5):
#     # 使用 numpy.histogram 分箱
#     max_length = max(data)
#     bins = np.arange(0, max_length + bin_size, bin_size)
#     hist, bin_edges = np.histogram(data, bins=bins)

#     # 绘制柱状图
#     plt.figure(figsize=(10, 6))
#     plt.bar(bin_edges[:-1], hist, width=bin_size, color='skyblue', align='edge')

#     plt.xlabel('Relevant Window Length Range')
#     plt.ylabel('Count')
#     plt.title('Distribution of Relevant Window Lengths (Binned)')
#     plt.xticks(bin_edges)  # 显示每个区间的起始点作为刻度
#     plt.savefig("/mnt/data/jiaqi/lighthouse/images/relevant_windows_tacos_train.png")
#     plt.show()

# if __name__ == "__main__":
#     # 假设 jsonl 文件名为 "data.jsonl"
#     file_path = "/mnt/data/jiaqi/online-vg/data/tacos/meta/train.jsonl"
#     relevant_windows_lengths = read_jsonl(file_path)
    
#     # 以5为一个区间大小绘制直方图
#     plot_histogram_binned(relevant_windows_lengths, bin_size=4)

import json
import matplotlib.pyplot as plt
import numpy as np

def read_jsonl(file_path):
    relevant_windows_lengths = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 读取 relevant_windows 并计算每个窗口的长度
            relevant_windows = data.get("relevant_windows", [])
            for window in relevant_windows:
                length = window[1] - window[0]  # 计算窗口长度
                relevant_windows_lengths.append(length)
    return relevant_windows_lengths

def plot_histogram_binned(data, bin_size=5, output_file=None):
    # 使用 numpy.histogram 分箱
    max_length = max(data)
    bins = np.arange(0, max_length + bin_size, bin_size)
    hist, bin_edges = np.histogram(data, bins=bins)

    # 计算累积的比例
    total_count = sum(hist)
    cumulative_hist = np.cumsum(hist)  # 计算累积的直方图
    cumulative_proportions = cumulative_hist / total_count  # 每个累积区间的比例

    # 如果指定了输出文件路径，则写入文件
    if output_file:
        with open(output_file, 'w') as f:
            for i in range(1, len(bin_edges)):  # 修改为累积区间
                range_start = 0
                range_end = bin_edges[i]
                proportion = cumulative_proportions[i-1]  # 累积比例
                f.write(f"{range_start}-{int(range_end)}: {proportion:.4f}\n")

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=bin_size, color='skyblue', align='edge')

    plt.xlabel('Relevant Window Length Range')
    plt.ylabel('Count')
    plt.title('Distribution of Relevant Window Lengths (Binned)')
    plt.xticks(bin_edges)  # 显示每个区间的起始点作为刻度
    plt.savefig("/mnt/data/jiaqi/lighthouse/images/relevant_windows_tacos_train.png")
    plt.show()

if __name__ == "__main__":
    # 假设 jsonl 文件名为 "data.jsonl"
    file_path = "/mnt/data/jiaqi/online-vg/data/tacos/meta/train.jsonl"
    relevant_windows_lengths = read_jsonl(file_path)
    
    # 输出比例文件路径
    output_file = "/mnt/data/jiaqi/lighthouse/images/relevant_windows_tacos_train.txt"
    
    # 以4为一个区间大小绘制直方图，并输出累积比例到文件
    plot_histogram_binned(relevant_windows_lengths, bin_size=2, output_file=output_file)
