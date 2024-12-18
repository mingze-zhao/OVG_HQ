#!/usr/bin/env python3
import os

# 在此显式定义参数
DIRECTORY = "/data-store/zengrh/jiaqi/online-vg/lighthouse_reproduce/configs/qvhighlight"  # 替换为实际目录路径
OLD_STR = "/mnt/data"  # 替换为需要替换的旧字符串
NEW_STR = "/data-store/zengrh"  # 替换为需要替换的新字符串

def replace_in_yml_files(directory, old_str, new_str):
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 只处理yml文件
            if filename.endswith('.yml'):
                filepath = os.path.join(root, filename)
                # 读取文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 替换字符串
                content = content.replace(old_str, new_str)
                # 写回文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == "__main__":
    replace_in_yml_files(DIRECTORY, OLD_STR, NEW_STR)
    print("替换完成！")
