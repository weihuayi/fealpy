from ..nodetype import CNodeType, PortConf, DataType


class ParticleGenerationSPH(CNodeType):
    r"""Dam Break particle generator for SPH simulations.

    This node generates initial particle configurations for dam break simulations
    using the Smoothed Particle Hydrodynamics (SPH) method.

    Inputs:
        dx (float): Horizontal spacing between particles.
        dy (float): Vertical spacing between particles.

    Outputs:
        pp (tensor): Coordinates of fluid particles representing the water column.
        bpp (tensor): Coordinates of boundary particles forming the container walls.
    """

    TITLE: str = "溃坝问题粒子生成"
    PATH: str = "流体.粒子生成"
    DESC: str = (
        """这个节点用于生成溃坝模拟的初始设置，包括流体粒子和边界粒子的位置。
        流体粒子集中在一个矩形区域（模拟水坝内的水），边界粒子构成一个容器（底部、左侧和右侧的墙壁）。
        通过指定两个方向粒子间距dx和dy，可以生成相应的粒子分布。"""
    )
    INPUT_SLOTS = [
        PortConf("dx", dtype=DataType.FLOAT, ttype=1, title="水平粒子间隔"),
        PortConf("dy", dtype=DataType.FLOAT, ttype=1, title="垂直粒子间隔"),
    ]
    OUTPUT_SLOTS = [
        PortConf("dx", dtype=DataType.FLOAT, title="水平粒子间隔"),
        PortConf("dy", dtype=DataType.FLOAT, title="垂直粒子间隔"),
        PortConf("pp", dtype=DataType.TENSOR, title="流体粒子坐标"),
        PortConf("bpp", dtype=DataType.TENSOR, title="边界粒子坐标"),
    ]

    @staticmethod
    def run(dx, dy):
        from fealpy.mesh.node_mesh import BamBreak

        domain = BamBreak.from_bam_break_domain(dx, dy)
        pp = domain.node
        bpp = domain.nodedata
        return pp, bpp
