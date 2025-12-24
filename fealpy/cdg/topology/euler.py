from fealpy.mesh import TriangleMesh


class TopologyInvariant:
    """
    计算网格的拓扑不变量。
    """

    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh

    def euler_characteristic(self) -> int:
        """
        计算欧拉示性数 (Euler Characteristic).
        Chi = V - E + F
        """
        V = self.mesh.number_of_nodes()
        E = self.mesh.number_of_edges()
        F = self.mesh.number_of_cells()
        return V - E + F

    def genus(self) -> int:
        """
        计算亏格 (Genus).
        对于闭曲面: g = (2 - Chi) / 2
        对于带边界曲面: g = (2 - Chi - B) / 2
        """
        chi = self.euler_characteristic()

        # 获取边界圈的数量
        # 这里为了避免循环引用，简单判断边界边
        n_bd_edges = self.mesh.boundary_edge_flag().sum()

        # 如果没有边界边，则是闭曲面
        if n_bd_edges == 0:
            return int((2 - chi) / 2)
        else:
            # 注意：对于带边界曲面，准确计算 genus 需要知道边界圈数 B
            # 这里暂时简化，通常 CCG 处理的是同胚于圆盘的 (g=0)
            # 如果需要严格计算，需结合 BoundaryProcessor
            return int(1 - chi / 2)  # 仅作参考，建议结合 BoundaryProcessor