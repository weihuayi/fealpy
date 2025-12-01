from typing import Union, List, Dict, Any, Tuple, Optional, Callable
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["YeeUniformMesh"]

class YeeUniformMesh(CNodeType):
    r"""Yee grid initializer for electromagnetic field simulations.

    Inputs:
        domain (list): Computational domain [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].
        n (int): Number of grid points in each direction.
        Ex_init_func (callable): Electric field x-component initialization function.
        Ey_init_func (callable): Electric field y-component initialization function.
        Ez_init_func (callable): Electric field z-component initialization function.
        Hx_init_func (callable): Magnetic field x-component initialization function.
        Hy_init_func (callable): Magnetic field y-component initialization function.
        Hz_init_func (callable): Magnetic field z-component initialization function.
        dt (float): Time step size for time-dependent initialization.
    
    Outputs:
        mesh (object): YeeUniformMesher instance.
        E_fields (dict): Initial electric field components.
        H_fields (dict): Initial magnetic field components.
    """
    TITLE: str = "Yee网格初始化器"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点创建Yee网格并初始化电磁场分量，考虑电场和磁场在Yee网格上的交错分布特性。"""
    
    INPUT_SLOTS = [
        PortConf("domain", DataType.DOMAIN, 1, title="计算域"),
        PortConf("n", DataType.INT, 0, title="剖分数", default=50),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="Yee网格实例"),

    ]

    @staticmethod
    def run(domain: List[float], n: int,) -> tuple:
        from fealpy.cem.mesh.yee_uniform_mesher import YeeUniformMesher
        
        # 创建Yee网格
        if len(domain) == 4:  # 2D: [xmin, xmax, ymin, ymax]
            mesher = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=0)
        else:  # 3D: [xmin, xmax, ymin, ymax, zmin, zmax]
            mesher = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=n)

        return mesher