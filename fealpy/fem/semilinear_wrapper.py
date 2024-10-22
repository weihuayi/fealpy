from functools import partial
from typing import Union, Callable, Optional, Any, TypeVar, Tuple, Dict

from ..typing import TensorLike, CoefLike
from ..functionspace.space import FunctionSpace as _FS
from .integrator import SemilinearInt, LinearInt
from ..functional import linear_integral, get_semilinear_coef
from ..utils import is_scalar, is_tensor, fill_axis, process_coef_func
from ..backend import backend_manager as bm


class SemilinearWrapperInt(SemilinearInt):
    """### Semilinear Wrapper Integrator
    A wrapper class that converts a LinearInt into a SemilinearInt by extracting and passing parameters."""

    def __init__(self, linear_int: LinearInt, method: Optional[str]=None):
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.linear_int = linear_int

    def assembly(self, space):
        uh = self.linear_int.coef.uh
        coef = self.linear_int.coef 
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.linear_int.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if bm.backend_name =='numpy' and self.linear_int.coef.grad_kernel_func is not None:
            val_F = -uh.grad_value(bcs)                    # [NC, NQ, dof_numel]
            coef_F = get_semilinear_coef(val_F, coef)
            F = linear_integral(gphi, ws, cm, coef_F, batched=self.linear_int.batched)
        else:
            uh_ = uh[space.cell_to_dof()]
            F = self.auto_grad(space, uh_, coef, batched=self.linear_int.batched)

        return self.linear_int.assembly(space), F

    def cell_integral(self, u, gphi, cm, ws, coef, batched) -> TensorLike:
        val = bm.einsum('i, qid -> qd', u, gphi)

        if coef is None:
            return bm.einsum('q, qid, qd -> i', ws, gphi, val) * cm

        if is_scalar(coef):
            return bm.einsum('q, qid, qd -> i', ws, gphi, val) * cm * coef

        if is_tensor(coef):
            coef = fill_axis(coef, 3 if batched else 2)
            print(bm.einsum(f'q, qid, qd, ...qd -> ...i', ws, gphi, val, coef) * cm)
            return bm.einsum(f'q, qid, qd, ...qd -> ...i', ws, gphi, val, coef) * cm

    def auto_grad(self, space, uh_, coef, batched) -> TensorLike:
        _, ws, gphi, cm, _ = self.linear_int.fetch(space)
        fn_F = bm.vmap(
            partial(self.cell_integral, ws=ws, coef=coef, batched=batched)
        )
        return -fn_F(uh_, gphi, cm)
