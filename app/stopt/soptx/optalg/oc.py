from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike, Tuple

from builtins import float

class OC:
    def __init__(self, dce, dve):
        """
        Initialize the OC method.

        Args:
            dce: TensorLike, compliance sensitivity.
            dve: TensorLike, volume sensitivity.
        """
        self.dce = dce
        self.dve = dve

    def update_design_variables(self, rho: TensorLike, g) -> Tuple[TensorLike, float]:
        """
        Update the design variables using the OC method.

        Args:
            rho: TensorLike, current design variables.
            g: float, current volume constraint value.

        Returns:
            tuple: Updated design variables and volume constraint value.
        """
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        rho_new = bm.zeros_like(rho, dtype=rho.dtype)

        while (l2 - l1) / (l2 + l1) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            rho_new[:] = bm.maximum(0.0, bm.maximum(rho - move, 
                    bm.minimum(1.0, bm.minimum(rho + move, rho * bm.sqrt(-self.dce / self.dve / lmid)))))
            gt = g + bm.sum(self.dve * (rho_new - rho))
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid

        return rho_new, gt
