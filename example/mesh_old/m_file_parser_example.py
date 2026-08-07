import argparse
import sys
from pathlib import Path

from fealpy.backend import bm
from fealpy.mesh import TriangleMesh
from fealpy.mesh import MFileParser

parser = argparse.ArgumentParser(
    description="计算共形几何工具：将 .m 网格文件解析并转换为 .vtu 格式。"
)

parser.add_argument(
    "--input_file",
    default="../../data/girl.m",
    type=str,
    help="输入的 .m 文件路径"
)

parser.add_argument(
    "--output", "-o",
    type=str,
    default='./data/girl',
    help="导出的 .vtu 文件路径 (默认为空，即不导出)"
)

args = parser.parse_args()

input_path = Path(args.input_file)
if not input_path.exists():
    print(f"错误: 找不到输入文件 '{input_path}'")
    sys.exit(1)

print(f"正在读取文件: {input_path} ...")

file_parser = MFileParser()
parser = file_parser.parse(str(input_path))

mesh = parser.to_mesh(TriangleMesh)
mesh_uv = TriangleMesh(mesh.nodedata['uv'], mesh.cell)
print("节点数:", mesh.number_of_nodes())
print("单元数:", mesh.number_of_cells())


print("解析成功！")

if args.output:
    output_path = Path(args.output)

    # 确保输出目录存在
    if not output_path.parent.exists():
        print(f"创建输出目录: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在导出到: {output_path} ...")
    try:
        # 导出为 VTU (Paraview 可读)
        mesh.to_vtk(fname=str(output_path)+"_original.vtu")
        mesh_uv.to_vtk(fname=str(output_path)+"_uv.vtu")
        print("导出完成。")
    except Exception as e:
        print(f"导出文件时发生错误: {e}")
else:
    print("未指定输出路径，跳过导出步骤。")

