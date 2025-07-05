import meshio


# 使用 meshio 转换为 VTU 格式
def convert_msh_to_vtu(msh_file, vtu_file):
    mesh = meshio.read(msh_file)

    # 提取四面体单元（兼容不同版本的Gmsh输出）
    tetra_cells = []
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":  # 确认单元类型名称
            tetra_cells.append(cell_block)

    if not tetra_cells:
        raise ValueError("未找到四面体单元，请检查网格生成设置")

    # 处理可能缺失的物理组数据
    cell_data = {}
    if "gmsh:physical" in mesh.cell_data:
        # 对齐物理组数据与单元类型
        physical_data = []
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra":
                if i < len(mesh.cell_data["gmsh:physical"]):
                    physical_data.append(mesh.cell_data["gmsh:physical"][i])
        if physical_data:
            cell_data["physical"] = physical_data

    # 创建新网格对象
    tetra_mesh = meshio.Mesh(
        points=mesh.points,
        cells=tetra_cells,
        cell_data=cell_data,
        point_data=mesh.point_data
    )

    tetra_mesh.write(vtu_file)


if __name__ == "__main__":
    # 执行格式转换
    convert_msh_to_vtu("metalenses_0619_2130_lc0.1.msh", "metalenses_0619_2130_lc0.1.vtu")

    print("VTU 文件生成完成，包含完整的四面体网格信息！")
