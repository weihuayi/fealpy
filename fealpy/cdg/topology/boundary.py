from typing import List, Tuple
from fealpy.backend import bm
from fealpy.mesh import TriangleMesh


class BoundaryProcessor:
    """
    边界处理类。
    职责：
    1. 识别并排序边界节点 (支持多个边界圈)。
    2. 生成边界参数化坐标 (如映射到单位圆)。
    """

    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh
        self._ordered_loops = None  # 缓存计算结果

    def get_boundaries(self) -> List[bm.ndarray]:
        """
        提取所有边界圈，并按连接顺序排序。

        Returns:
            List[bm.ndarray]: 一个列表，包含多个 numpy 数组。
                              每个数组是一个独立的边界圈节点索引 [v1, v2, v3, ... v1]
        """
        if self._ordered_loops is not None:
            return self._ordered_loops

        # 1. 获取所有边界边
        edge = self.mesh.ds.edge
        is_bd = self.mesh.boundary_edge_flag()
        bd_edges = edge[is_bd]  # (N_be, 2)

        if len(bd_edges) == 0:
            return []

        # 2. 构建邻接表: node -> [neighbor_node, ...]
        adj = {}
        for u, v in bd_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        # 3. 遍历提取循环
        visited = set()
        loops = []

        # 获取所有边界点
        bd_nodes = list(adj.keys())

        for start_node in bd_nodes:
            if start_node in visited:
                continue

            # 开始一个新的循环遍历
            current_loop = [start_node]
            visited.add(start_node)

            curr = start_node
            # 找下一个点
            # 边界通常是闭合的 loop，或者是一条线（如果不闭合）
            # CCG 中通常假设边界是闭合圈
            while True:
                neighbors = adj[curr]
                # 找一个没被加入当前路径的邻居，或者如果是终点则闭合
                next_node = None
                for n in neighbors:
                    # 如果 n 是起点且路径长度>2，说明闭合了
                    if n == start_node and len(current_loop) > 2:
                        next_node = n  # 闭合
                        break
                    # 否则找一个没访问过的
                    if n not in visited:
                        next_node = n
                        break

                if next_node is None:
                    # 可能是因为回到了起点 (在上面的 break 处理)
                    # 或者断开了 (非流形/非闭合边界)
                    break

                if next_node == start_node:
                    break

                current_loop.append(next_node)
                visited.add(next_node)
                curr = next_node

            loops.append(bm.array(current_loop, dtype=bm.int64))

        self._ordered_loops = loops
        return loops

    def map_to_circle(self, loop_index: int = 0) -> Tuple[bm.ndarray, bm.ndarray]:
        """
        将指定的边界圈映射到单位圆 (Arc-length Parameterization)。

        Args:
            loop_index (int): 处理第几个边界圈 (默认为 0，即最外层边界)。

        Returns:
            bd_idx (bm.ndarray): 边界节点全局索引。
            bd_uv (bm.ndarray): 对应的 (u, v) 坐标，形状 (N, 2)。
        """
        loops = self.get_boundaries()
        if loop_index >= len(loops):
            raise ValueError(f"Loop index {loop_index} out of range (Total: {len(loops)})")

        bd_idx = loops[loop_index]
        nodes = self.mesh.entity('node')

        # 1. 提取边界坐标
        coords = nodes[bd_idx]

        # 2. 计算弦长 (Arc Length)
        # next_coords 是错位后的坐标，用于计算 v[i] 到 v[i+1] 的距离
        next_coords = bm.roll(coords, -1, axis=0)
        dists = bm.linalg.norm(next_coords - coords, axis=1)

        total_len = bm.sum(dists)
        cum_len = bm.concatenate(([0], bm.cumsum(dists)))[:-1]  # 累积长度

        # 3. 映射到 [0, 2pi]
        theta = 2.0 * bm.pi * cum_len / total_len

        u = bm.cos(theta)
        v = bm.sin(theta)

        return bd_idx, bm.column_stack((u, v))

    def map_to_square(self):
        """
        [扩展接口] 将边界映射到正方形。
        你需要指定 4 个角点索引，然后分段进行线性插值。
        (留给你练习实现)
        """
        pass