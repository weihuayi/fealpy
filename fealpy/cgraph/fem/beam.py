from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Beam"]


class Beam(CNodeType):
    r"""Assemble the global stiffness matrix and load vector for an Euler-Bernoulli beam FEM model.

    Parameters (read from options):
        space: Lagrange function space (scalar or tensor function space).
        beam_E (float): Young's modulus of the beam.
        beam_nu (float): Poisson's ratio of the beam.
        I (float or tensor): Second moment of area (area moment of inertia).
        external_load / distributed_load (callable or array, optional): External distributed load, returning a global load vector matching DOFs.
        dirichlet_idx / dirichlet_dof_index (callable or array, optional): Indices of DOFs where Dirichlet boundary conditions are applied.
        beam_type (str, optional): Beam element type, e.g. "euler_bernoulli_2d", "normal_2d", "euler_bernoulli_3d", "normal_3d".
        Other options: Additional keys in the options dict may be used by integrators or material classes.

    Returns:
        K: Global stiffness matrix (sparse matrix or tensor).
        F: Global load vector.

    Notes:
        Uses fealpy's TensorFunctionSpace, BilinearForm, LinearForm and corresponding Euler-Bernoulli beam integrators
        to assemble the stiffness matrix and load vector. Inputs are obtained from the provided options dictionary.
    """
    TITLE: str = "欧拉梁有限元模型"
    PATH: str = "有限元.方程离散"
    DESC: str = "组装欧拉梁的刚度矩阵和载荷"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的泊松比"),
        PortConf("I", DataType.FLOAT, 1, desc="惯性矩",  title="惯性矩"),
        PortConf("distributed_load", DataType.FLOAT, 1, desc="分布载荷", title="分布载荷"),
        PortConf("beam_type", DataType.MENU, 0, desc="梁单元选择",title="梁单元类型", default="euler_bernoulli_2d",
                 items=["euler_bernoulli_2d", "normal_2d", "euler_bernoulli_3d", "normal_3d"]),
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.TENSOR, desc="刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="载荷向量",  title="全局载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.fem import BilinearForm, LinearForm
        from fealpy.csm.fem import EulerBernoulliBeamDiffusionIntegrator
        from fealpy.csm.fem import EulerBernoulliBeamSourceIntegrator

        class BeamMaterial:
            def __init__(self, options: dict):
                self.E  = options.get("beam_E")
                self.nu = options.get("beam_nu")
                self.I = options.get("I")
                
        
        beam_material = BeamMaterial(options=options) 
        
        
        space = options.get("space")
        distributed_load = options.get("distributed_load")
        beam_type = options.get("beam_type")
        
        tspace = TensorFunctionSpace(space, shape=(-1, 2))
        l = tspace.mesh.cell_length()
        beam_integrator = EulerBernoulliBeamDiffusionIntegrator(
            tspace, beam_type, material=beam_material)
        bform = BilinearForm(tspace)
        bform.add_integrator(beam_integrator)
        K = bform.assembly()
        
        lform = LinearForm(tspace)
        FF  = EulerBernoulliBeamSourceIntegrator(tspace, beam_type, source=-distributed_load, l=l)
        lform.add_integrator(FF)
        F = lform.assembly()
        
        return K, F