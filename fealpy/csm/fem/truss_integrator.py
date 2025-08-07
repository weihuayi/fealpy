
from fealpy.backend import backend_manager as bm
from fealpy.typing import Optional, Literal, TensorLike, Index, _S
from fealpy.decorator import variantmethod

from fealpy.functionspace.tensor_space import TensorFunctionSpace as _TS
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt, enable_cache)

class TrussIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, 
                 space, truss_type, E, l, A=None,
                 method: Optional[str]=None) -> None:
        """
        Parameters:
            space: FE space
            beam_type: 'euler_bernoulli_2d', 'normal_2d', or 'euler_bernoulli_3d'
            E: Young's modulus
            l: element length array (NC,)
            A: cross-sectional area (used for normal_2d/euler_bernoulli_3d)
            I: moment of inertia (used for euler_bernoulli_2d/normal_2d)
            Iy, Iz: moments of inertia (used for euler_bernoulli_3d)
            G: shear modulus (used for euler_bernoulli_3d)
            J: torsional constant (used for euler_bernoulli_3d)
        """
        self.space = space
        self.type = truss_type.lower()
        self.E = E
        self.l = l
        self.A = A
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        #if self.type == 'bar_1d'
            
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()
    
    @variantmethod("bar_1d")
    def assembly(self, space: _TS) -> TensorLike:
        E = self.E
        l = self.l
        A = self.A
        coef0 = E*A[0] / l
        coef1 = E*A[1] / l
        coef2 = E*A[2] / l

        k00 = coef0
        k01 = -coef0

        k11 = coef0 + coef1
        k12 = -coef1

        k22 = coef1 + coef2
        k23 = -coef2

        k33 = coef2
        
        K = bm.stack([
        bm.stack([k00, k01, 0.0, .0], axis=-1),  # 第 0 行
        bm.stack([k01, k11, k12, .0], axis=-1),  # 第 1 行
        bm.stack([0.0, k12, k22, k23], axis=-1),  # 第 2 行
        bm.stack([0.0, 0.0, k23, k33], axis=-1),  # 第 3 行
    ], axis=1)
        
        return K


       