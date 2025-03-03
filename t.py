import os
import re

# 定义文件夹路径
folder_path = "/root/FM4TS/scripts"

# 遍历文件夹中的所有.sh文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".sh"):  # 确保处理的是.sh文件
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(file_path, 'w') as file:
                for line in lines:
                    # 删除包含"batch_size": ?的部分
                    line = re.sub(r', "batch_size": \d+', '', line)
                    file.write(line)
