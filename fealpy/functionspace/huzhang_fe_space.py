from ..mesh import Mesh

class HuZhangFESpace:
    """Factory class for creating HuZhang finite element spaces."""
    def __new__(cls, mesh: Mesh, p: int = 1, ctype: str = 'C'):
        TD = mesh.top_dimension()
        
        if TD == 2:
            from .huzhang_fe_space_2d import HuZhangFESpace2d
            return HuZhangFESpace2d(mesh, p=p, ctype=ctype)
        elif TD == 3:
            from .huzhang_fe_space_3d import HuZhangFESpace3d
            return HuZhangFESpace3d(mesh, p=p, ctype=ctype)
        else:
            raise ValueError(f"Unsupported dimension: {TD}. Only 2D and 3D are supported.")
