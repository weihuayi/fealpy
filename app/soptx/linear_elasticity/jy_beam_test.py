import re

def parse_file(file_path):
    # 初始化数据结构
    nodes = []
    cells = []
    element_stiffness_matrices = {}
    material_params = {}
    cell_data = {}
    
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义各个部分的正则表达式模式
    solid_pattern = re.compile(r"Solid:\s*(.*?)\s*EndSolid", re.DOTALL)
    element_stiffness_pattern = re.compile(r"ElementEtiffnessMatrix:\s*(.*?)\s*EndElementEtiffnessMatrix", re.DOTALL)
    material_pattern = re.compile(r"Material:\s*(.*?)\s*EndMaterial", re.DOTALL)
    cell_data_pattern = re.compile(r"CellData:\s*(.*?)\s*EndCellData", re.DOTALL)
    
    # 解析 Solid 部分
    solid_match = solid_pattern.search(content)
    if solid_match:
        solid_content = solid_match.group(1)
        
        # 解析 Nodes
        nodes_section_match = re.search(r"Nodes:\s*(.*?)\s*(?:Cells:|$)", solid_content, re.DOTALL)
        if nodes_section_match:
            nodes_section = nodes_section_match.group(1)
            # 每行节点格式: <id>: (<x>, <y>, <z>)
            node_lines = nodes_section.strip().split('\n')
            for line in node_lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\d+:\s*\(([^)]+)\)', line)
                if match:
                    coords = tuple(map(float, match.group(1).split(',')))
                    nodes.append(coords)
        
        # 解析 Cells
        cells_section_match = re.search(r"Cells:\s*(.*?)\s*$", solid_content, re.DOTALL)
        if cells_section_match:
            cells_section = cells_section_match.group(1)
            # 每行单元格式: <id>: (<n1>, <n2>, <n3>, <n4>)
            cell_lines = cells_section.strip().split('\n')
            for line in cell_lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\d+:\s*\(([^)]+)\)', line)
                if match:
                    node_ids = tuple(map(int, match.group(1).split(',')))
                    cells.append(node_ids)
    
    # 解析 ElementEtiffnessMatrix 部分
    element_stiffness_match = element_stiffness_pattern.search(content)
    if element_stiffness_match:
        element_stiffness_content = element_stiffness_match.group(1)
        # 将内容按行分割
        lines = element_stiffness_content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Cell"):
                match = re.match(r"Cell\s+(\d+)", line)
                if match:
                    cell_id = int(match.group(1))
                    i += 1
                    matrix = []
                    # 读取接下来的12行作为矩阵
                    for _ in range(12):
                        if i >= len(lines):
                            break
                        matrix_line = lines[i].strip()
                        if not matrix_line:
                            i += 1
                            continue
                        # 提取所有浮点数，包括科学计数法
                        numbers = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', matrix_line)))
                        if numbers:
                            matrix.append(numbers)
                        i += 1
                    element_stiffness_matrices[cell_id] = matrix
            else:
                i += 1
    
    # 解析 Material 部分
    material_match = material_pattern.search(content)
    if material_match:
        material_content = material_match.group(1)
        # 解析材料参数，例如: E = 30000
        lines = material_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'(E|nu|Lam|mu)\s*=\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', line)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                material_params[key] = value
    
    # 解析 CellData 部分
    cell_data_match = cell_data_pattern.search(content)
    if cell_data_match:
        cell_data_content = cell_data_match.group(1)
        # 按 'Cell <id>' 分割
        cell_sections = re.split(r"Cell\s+(\d+)", cell_data_content)
        # split 结果: [前导内容, id, 内容, id, 内容, ...]
        for j in range(1, len(cell_sections), 2):
            cell_id = int(cell_sections[j])
            cell_content = cell_sections[j+1]
            cell_data[cell_id] = {}
            
            # 解析 Grad Lambda
            grad_lambda_match = re.search(r"Grad Lambda\s*(.*?)\s*(Displacement\s*:|Strain\s*:|Stress\s*:|$)", cell_content, re.DOTALL)
            if grad_lambda_match:
                grad_lambda_content = grad_lambda_match.group(1).strip()
                grad_lambda_lines = grad_lambda_content.split('\n')
                grad_lambda = []
                for g_line in grad_lambda_lines:
                    g_line = g_line.strip()
                    if not g_line:
                        continue
                    grad = list(map(float, g_line.split()))
                    grad_lambda.append(grad)
                cell_data[cell_id]['Grad Lambda'] = grad_lambda
            
            # 解析 Displacement
            displacement_match = re.search(r"Displacement\s*:\s*(.*?)\s*(Strain\s*:|Stress\s*:|Grad Lambda|$)", cell_content, re.DOTALL)
            if displacement_match:
                displacement_content = displacement_match.group(1).strip()
                displacement_lines = displacement_content.split('\n')
                displacement = []
                for d_line in displacement_lines:
                    d_line = d_line.strip()
                    if not d_line:
                        continue
                    disp = list(map(float, d_line.split()))
                    displacement.append(disp)
                cell_data[cell_id]['Displacement'] = displacement
            
            # 解析 Strain
            strain_match = re.search(r"Strain\s*:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?(?:\s+[-+]?\d*\.?\d+(?:e[-+]?\d+)?)*)", cell_content)
            if strain_match:
                strain_str = strain_match.group(1).strip()
                strain = list(map(float, strain_str.split()))
                cell_data[cell_id]['Strain'] = strain
            
            # 解析 Stress
            stress_match = re.search(r"Stress\s*:\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?(?:\s+[-+]?\d*\.?\d+(?:e[-+]?\d+)?)*)", cell_content)
            if stress_match:
                stress_str = stress_match.group(1).strip()
                stress = list(map(float, stress_str.split()))
                cell_data[cell_id]['Stress'] = stress
    
    return {
        'nodes': nodes,
        'cells': cells,
        'element_stiffness_matrices': element_stiffness_matrices,
        'material_params': material_params,
        'cell_data': cell_data
    }

if __name__ == "__main__":
    # 替换为您的txt文件路径
    file_path = '/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/log.txt'
    data = parse_file(file_path)
    
    # # 打印提取的数据
    # print("Nodes:")
    # for idx, coords in enumerate(data['nodes']):
    #     print(f"Node {idx}: {coords}")
    
    # print("\nCells:")
    # for idx, node_ids in enumerate(data['cells']):
    #     print(f"Cell {idx}: {node_ids}")
    
    # print("\nMaterial Parameters:")
    # for param, value in data['material_params'].items():
    #     print(f"{param}: {value}")
    
    # print("\nElement Stiffness Matrices:")
    # for cell_id in sorted(data['element_stiffness_matrices']):
    #     print(f"\nCell {cell_id}:")
    #     for row in data['element_stiffness_matrices'][cell_id]:
    #         print(row)
    # print("\nCell Data:")
    # for cell_id in sorted(data['cell_data']):
    #     print(f"\nCell {cell_id}:")
    #     cdata = data['cell_data'][cell_id]
    #     for key, value in cdata.items():
    #         print(f"  {key}:")
    #         if isinstance(value, list):
    #             for sublist in value:
    #                 print(f"    {sublist}")
    #         else:
    #             print(f"    {value}")
    print("--------------------")

    nodes = data['nodes']
    cells = data['cells']
    element_stiffness_matrices = data['element_stiffness_matrices']
    material_params = data['material_params']
    cell_data = data['cell_data']

    NN = len(nodes)
    NC = len(cells)

    import numpy as np
    real_nodes = np.zeros((NN, 3), dtype=np.float64)
    real_cells = np.zeros((NC, 4), dtype=np.int32)
    for i, node in enumerate(nodes):
        real_nodes[i] = np.array(node)
    for i, cell in enumerate(cells):
        real_cells[i] = np.array(cell)

    K_real_elem = np.zeros((NC, 12, 12), dtype=np.float64)
    for i in range(NC):
        K_real_elem[i] = np.array(element_stiffness_matrices[i])

    E = material_params['E']
    nu = material_params['nu']
    Lam = material_params['Lam']
    mu = material_params['mu']

    grad_lambda_elem = np.zeros((NC, 4, 3), dtype=np.float64)
    displacement_elem = np.zeros((NC, 4, 3), dtype=np.float64)
    strain_elem = np.zeros((NC, 6), dtype=np.float64)
    stress_elem = np.zeros((NC, 6), dtype=np.float64)

    for i in range(NC):
        grad_lambda_elem[i] = np.array(cell_data[i]['Grad Lambda'])
        displacement_elem[i] = np.array(cell_data[i]['Displacement'])
        strain_elem[i] = np.array(cell_data[i]['Strain'])
        stress_elem[i] = np.array(cell_data[i]['Stress'])


    print("real_nodes:")
    print(real_nodes)
    print("real_cells:")
    print(real_cells)
    print("K_real_elem:")
    print(K_real_elem)
    print("E:")
    print(E)
    print("nu:")
    print(nu)
    print("Lam:")
    print(Lam)
    print("mu:")
    print(mu)
    print("grad_lambda_elem:")
    print(grad_lambda_elem)
    print("displacement_elem:")
    print(displacement_elem)
    print("strain_elem:")
    print(strain_elem)
    print("stress_elem:")
    print(stress_elem)

    print(-1)
