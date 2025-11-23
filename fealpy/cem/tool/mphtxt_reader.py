def read_mphtxt_mesh(filename):
    nodes = []
    elements = []
    entity_indices = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    def extract_block(start_marker):
        """提取指定起始标记后的非空行块"""
        start = None
        for i, line in enumerate(lines):
            if start_marker in line:
                start = i + 1
                break
        if start is None:
            return []
        block = []
        for line in lines[start:]:
            line = line.strip()
            if not line:
                break
            block.append(line)
        return block

    # 读取节点坐标
    node_lines = extract_block("# Mesh vertex coordinates")
    for line in node_lines:
        coords = list(map(float, line.split()))
        nodes.append(coords)

    # 读取单元（元素）拓扑结构
    elem_lines = extract_block("# Elements")
    for line in elem_lines:
        indices = list(map(int, line.split()))
        elements.append(indices)

    # 读取单元所属的几何区域编号
    region_lines = extract_block("# Geometric entity indices")
    for line in region_lines:
        entity_indices.append(int(line.strip()))

    return {
        "nodes": nodes,
        "elements": elements,
        "entity_indices": entity_indices
    }

# ✅ 用法示例
if __name__ == "__main__":
    from fealpy.backend import  backend_manager as bm
    from fealpy.mesh import TetrahedronMesh
    filename = "../data/metalenses_ele.mphtxt"  # 替换为实际文件路径
    mesh_data = read_mphtxt_mesh(filename)

    node = bm.array(mesh_data['nodes'], dtype=bm.float64)
    print(f"读取节点数: {len(node)}")
    cell = bm.array(mesh_data['elements'], dtype=bm.int64)
    print(f"读取单元数: {len(cell)}")
    cell_domain = bm.array(mesh_data['entity_indices'], dtype=bm.int32)
    print(f"区域编号数: {len(cell_domain)}")

    tet_mesh = TetrahedronMesh(node, cell)
    tet_mesh.celldata['domain'] = cell_domain
    tet_mesh.to_vtk("../data/metalenses_ele.vtu")

    print(-1)



