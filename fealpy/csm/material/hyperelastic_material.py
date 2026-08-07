from typing import Optional
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm


class HyperElasticMaterial:
    """
    Nearly incompressible Yeoh hyperelastic material.

    Strain energy density:

        W = W_iso + W_vol

    Isochoric part:

        W_iso =
            C10 * (I1_bar - 3)
          + C20 * (I1_bar - 3)^2
          + C30 * (I1_bar - 3)^3

    Volumetric part:

        W_vol =
            (1 / D1) * (J - 1)^2
    """

    def __init__(
        self,
        C10: float,
        C20: float = 0.0,
        C30: float = 0.0,
        D1: float = 1.0e-3,
    ) -> None:

        self.C10 = C10
        self.C20 = C20
        self.C30 = C30

        self.D1 = D1

    # =====================================================
    # Invariants
    # =====================================================
    def compute_C(self, F: TensorLike) -> TensorLike:
        """
        Right Cauchy-Green tensor

            C = F^T F

        Parameters
        ----------
        F : (NC, NQ, GD, GD)

        Returns
        -------
        C : (NC, NQ, GD, GD)
        """
        return bm.einsum('...ji,...jk->...ik', F, F)

    def compute_I1(self, F: TensorLike) -> TensorLike:
        """
        First invariant

            I1 = tr(C)

        Returns
        -------
        I1 : (NC, NQ)
        """
        C = self.compute_C(F)
        return bm.einsum('...ii->...', C)

    def compute_J(self, F: TensorLike) -> TensorLike:
        """
        Jacobian

            J = det(F)

        Returns
        -------
        J : (NC, NQ)
        """
        return bm.linalg.det(F)

    def compute_I1_bar(self, F: TensorLike) -> TensorLike:
        """
        Modified first invariant

            I1_bar = J^(-2/3) * I1

        Returns
        -------
        I1_bar : (NC, NQ)
        """
        I1 = self.compute_I1(F)
        J = self.compute_J(F)

        return bm.power(J, -2.0 / 3.0) * I1

    # =====================================================
    # Energy
    # =====================================================

    def strain_energy_density(self, F: TensorLike) -> TensorLike:
        """
        Total strain energy density.

        Returns
        -------
        W : (NC, NQ)
        """

        I1_bar = self.compute_I1_bar(F)
        J = self.compute_J(F)

        x = I1_bar - 3.0

        W_iso = (
            self.C10 * x
            + self.C20 * x * x
            + self.C30 * x * x * x
        )

        W_vol = (1.0 / self.D1) * (J - 1.0) ** 2

        return W_iso + W_vol

    # =====================================================
    # Energy derivatives
    # =====================================================

    def compute_dWdI1(self, I1_bar: TensorLike) -> TensorLike:
        """
        dW / d(I1_bar)

        Returns
        -------
        (NC, NQ)
        """

        x = I1_bar - 3.0

        return (
            self.C10
            + 2.0 * self.C20 * x
            + 3.0 * self.C30 * x * x
        )

    def compute_dWdJ(self, J: TensorLike) -> TensorLike:
        """
        dW_vol / dJ

        Returns
        -------
        (NC, NQ)
        """

        return (2.0 / self.D1) * (J - 1.0)

    # =====================================================
    # Constitutive response
    # =====================================================

    def stress(self, F: TensorLike) -> TensorLike:
        """
        First Piola-Kirchhoff stress.

        Parameters
        ----------
        F : (NC, NQ, GD, GD)

        Returns
        -------
        P : (NC, NQ, GD, GD)
        """

        I1 = self.compute_I1(F)

        J = self.compute_J(F)

        I1_bar = self.compute_I1_bar(F)

        dWdI1 = self.compute_dWdI1(I1_bar)

        dWdJ = self.compute_dWdJ(J)

        Finv = bm.linalg.inv(F)

        FinvT = bm.swapaxes(Finv, -1, -2)

        # ---------------------------------------------
        # Isochoric contribution
        # ---------------------------------------------

        term = (
            2.0 * F
            - (2.0 / 3.0)
            * I1[..., None, None]
            * FinvT
        )

        P_iso = (
            dWdI1[..., None, None]
            * bm.power(J[..., None, None], -2.0 / 3.0)
            * term
        )

        # ---------------------------------------------
        # Volumetric contribution
        # ---------------------------------------------


        P_vol = (
            dWdJ[..., None, None]
            * J[..., None, None]
            * FinvT
        )

        return P_iso + P_vol
    def tangent(self, F):#遍历求导
        """
        Numerical consistent tangent

        Returns
        -------
        A : (NC, NQ, GD, GD, GD, GD)
        """

        h = 1e-8

        P0 = self.stress(F)

        GD = F.shape[-1]

        shape = F.shape[:-2] + (GD, GD, GD, GD)

        A = bm.zeros(shape, dtype=F.dtype)

        for k in range(GD): #遍历求导
            for l in range(GD):

                dF = bm.zeros_like(F)

                dF[..., k, l] = h

                P1 = self.stress(F + dF)

                dP = (P1 - P0) / h  

                A[..., :, :, k, l] = dP

        return A