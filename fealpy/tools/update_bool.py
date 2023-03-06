import os
import re

# 目录路径
directory = '/home/why/fealpy/example'

# 遍历目录下的所有文件
for subdir, dirs, files in os.walk(directory):
    for file in files:
        # 如果不是 Python 文件，跳过
        if not file.endswith('.py'):
            continue
        
        # 打开文件并读取内容
        filepath = os.path.join(subdir, file)
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 查找所有符合条件的字符串，并替换为带下划线的版本
        new_content = re.sub(r'np\.bool(?!\w)', r'np.bool_', content)
        
        # 如果有替换，更新文件内容并保存
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)

