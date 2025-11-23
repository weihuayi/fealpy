#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <文件路径> <行号>"
    echo "示例: $0 fealpy/mesh/triangle_mesh.py 1258"
    exit 1
fi

FILE="$1"
LINE="$2"

# 检查文件是否存在
if [ ! -f "$FILE" ]; then
    echo "错误: 文件 $FILE 不存在"
    exit 2
fi

# 获取 git blame 信息
git blame -L "$LINE","$LINE" --show-email "$FILE"

