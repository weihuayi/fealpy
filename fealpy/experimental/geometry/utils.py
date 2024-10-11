import numpy as np


def line_transformer(ori_line, start_point, end_point):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    ori_start_point = ori_line[0]
    ori_end_point = ori_line[-1]

    t_matrix = np.zeros((4, 4))
    t_matrix[0, 0:2] = ori_start_point
    t_matrix[1, 2:] = ori_start_point
    t_matrix[2, 0:2] = ori_end_point
    t_matrix[3, 2:] = ori_end_point
    f = np.concatenate([start_point.reshape(-1, 1), end_point.reshape(-1, 1)], axis=0)

    t = np.linalg.solve(t_matrix, f).reshape(2, 2)
    new_line = np.einsum('ij,nj->ni', t, ori_line)

    return new_line


def delta_angle_calculator(start_angle, end_angle, input_type='angle'):
    '''
        计算两向量夹角（小于 \pi 的夹角）
        :param start_angle: 向量一
        :param end_angle: 向量二
        :param input_type: 输入类型，角度或向量
        :return: 两向量角度变化（弧度制），结果区间为 [-\pi, \pi]，按逆时针方向旋转
        '''
    assert input_type in ['angle', 'vector']
    if isinstance(start_angle, list):
        start_angle = np.array(start_angle)
    if isinstance(end_angle, list):
        end_angle = np.array(end_angle)
    if input_type == 'vector':
        start_angle = np.arctan2(start_angle[1], start_angle[0])
        end_angle = np.arctan2(end_angle[1], end_angle[0])
    condition1 = (end_angle - start_angle) < -np.pi
    condition2 = (end_angle - start_angle) > np.pi

    result = np.where(condition1, end_angle - start_angle + 2 * np.pi,
                      np.where(condition2, end_angle - start_angle - 2 * np.pi, end_angle - start_angle))
    return result


def next_edge_searcher(current_edge, next_node, node_to_out_edge, total_line, rotation_direction='counter_clock'):
    """
    根据输入边与到达节点，确定下一条边
    :param current_edge: 当前边编号
    :param next_node: 当前边到达节点编号
    :param node_to_out_edge: 节点与该节点出发边的映射
    :param total_line: 所有边对应的几何线
    :param rotation_direction: 旋转方向，可选项为逆时针方向与顺时针方向，默认为逆时针方向，[1, 2, '1', '2', 'counter_clock', 'clock']
    :return: 下一条边对应的编号
    """
    assert rotation_direction in [1, 2, '1', '2', 'counter_clock', 'clock']

    next_edge = node_to_out_edge[next_node][0]
    input_vector = total_line[current_edge][-1] - total_line[current_edge][-2]
    if rotation_direction in [1, '1', 'counter_clock']:
        delta_angle = max(
            delta_angle_calculator(input_vector, total_line[next_edge][1] - total_line[next_edge][0], 'vector'), 0)
        max_delta_angle = delta_angle if abs(delta_angle - 3.14) > 0.01 else 0
        for e in node_to_out_edge[next_node][1:]:
            delta_angle = max(delta_angle_calculator(input_vector, total_line[e][1] - total_line[e][0], 'vector'), 0)
            delta_angle = delta_angle if abs(delta_angle - 3.14) > 0.01 else 0
            if delta_angle > max_delta_angle:
                next_edge = e
                max_delta_angle = delta_angle
    else:
        delta_angle = delta_angle_calculator(input_vector, total_line[next_edge][1] - total_line[next_edge][0],
                                             'vector')
        min_delta_angle = abs(delta_angle) if delta_angle <= 0 else 3.14
        for e in node_to_out_edge[next_node][1:]:
            delta_angle = delta_angle_calculator(input_vector, total_line[e][1] - total_line[e][0], 'vector')
            delta_angle = abs(delta_angle) if delta_angle <= 0 else 3.14
            if delta_angle < min_delta_angle:
                next_edge = e
                min_delta_angle = delta_angle
    return next_edge


def subdomain_divider(line: list, node: np.ndarray, edge: np.ndarray, boundary_edge: np.ndarray):
    """
    基于半边网格子区域自动划分方法
    :param line: 一组线段上所有点的坐标组成的列表，实际集合区域边界
    :param node: 各边界的交点, 要求第一个节点为边界节点
    :param edge: 各边界的起点与终点对应的 node 的索引, 要求边界边沿逆时针方向
    :param boundary_edge: 边界边编号数组
    :return : 半边数据结构的网格代表的子区域
    """
    half_edge = np.zeros((len(edge) * 2, 5), dtype=np.int_)
    half_edge[::2, 0] = edge[:, 1]
    half_edge[1::2, 0] = edge[:, 0]
    half_edge[::2, 4] = 2 * np.arange(len(edge)) + 1
    half_edge[1::2, 4] = 2 * np.arange(len(edge))
    cell_idx = 0
    # 此处限制了传入的edge与对应的line需要为逆时针，考虑优化这一限制
    boundary_edge = boundary_edge * 2 + 1
    boundary_edge = boundary_edge.tolist()

    total_line = []
    for l in line:
        total_line.append(l)
        total_line.append(l[::-1])

    total_edge = np.zeros((len(edge) * 2, 2), dtype=np.int_)
    total_edge[::2] = edge
    total_edge[1::2] = edge[:, ::-1]
    node_to_out_edge = []
    for n in range(len(node)):
        node_to_out_edge.append([])
    # 根究边起点，构造关于所有节点的 edge_out 列表
    for e in range(len(total_edge)):
        node_to_out_edge[total_edge[e, 0]].append(e)

    # 顺时针外边界处理
    bd_edge_find_flag = False
    node_idx = 0
    # 寻找第一条边界边，及其起始点
    for e_line in node_to_out_edge:
        for e in e_line:
            if e in boundary_edge:
                bd_edge_find_flag = True
                break
        if bd_edge_find_flag:
            break
        node_idx += 1
    # 遍历边界
    current_edge = e
    node_to_out_edge[node_idx].remove(current_edge)
    half_edge[current_edge, 1] = -1
    next_node = half_edge[current_edge, 0]
    while next_node != node_idx:
        for ne in node_to_out_edge[next_node]:
            if ne in boundary_edge:
                next_edge = ne
                break
        # next_edge = next_edge_searcher(current_edge, next_node, node_to_out_edge, total_line, rotation_direction='clock')
        node_to_out_edge[next_node].remove(next_edge)
        half_edge[current_edge, 2] = next_edge
        half_edge[next_edge, 3] = current_edge
        half_edge[next_edge, 1] = -1
        current_edge = next_edge
        next_node = half_edge[current_edge, 0]
    half_edge[current_edge, 2] = e
    half_edge[e, 3] = current_edge

    for n in range(len(node)):
        while node_to_out_edge[n]:
            first_edge = node_to_out_edge[n].pop()
            current_edge = first_edge
            half_edge[current_edge, 1] = cell_idx
            next_node = half_edge[current_edge, 0]

            while next_node != n:
                next_edge = next_edge_searcher(current_edge, next_node, node_to_out_edge, total_line)
                node_to_out_edge[next_node].remove(next_edge)
                half_edge[current_edge, 2] = next_edge
                half_edge[next_edge, 3] = current_edge
                half_edge[next_edge, 1] = cell_idx
                current_edge = next_edge
                next_node = half_edge[current_edge, 0]
            half_edge[current_edge, 2] = first_edge
            half_edge[first_edge, 3] = current_edge
            cell_idx += 1

    return half_edge

