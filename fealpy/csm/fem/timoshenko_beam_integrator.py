from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS

from ..material.timoshenko_beam_material import TimoshenkoBeamMaterial


class TimoshenkoBeamIntegrator(LinearInt, OpInt, CellInt):
    """
    Integrator for Timoshenko beam problems.
    """

    def __init__(self, 
                 space: _FS, 
                 beam_type: Literal['timo_2d', 'timo_3d'],
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.type = beam_type.lower()
        self.meterial = material
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    def _coord_transfrom(self):
        # TODO 
        pass
    
    def _bars_length(self):
        pass
    
    @variantmethod
    def assembly(self, D, R, LBM) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.
        This function computes the (12, 12) stiffness matrix for each element.
        Parameters:
            E : Young's modulus.
            G: shear modulus.
            AX : cross-sectional area. 
            Iy: moment of inertia about z-axis.
            Iz: moment of inertia about y-axis.
            Ix: torsional constant. 
            l: element length.
            FSY and FSZ: The shear correction factor, 6/5 for rectangular and 10/9 for circular.
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        Notes:
            The stiffness matrix is derived from Euler-Bernoulli beam theory and is suitable for small deformation linear elastic analysis.
        """
        E = self.material.E
        nu = self.material.nu

        AX, AY, AZ = self.material.cal_A()
        Iy, Iz, Ix = self.material.cal_I()
        FSY, FSZ = self.material.shear_factor()

        FY = 12 * E * Iz / G / AY / (LBM**2)  # Phi_y
        FZ = 12 * E * Iy / G / AZ / (LBM**2)  # Phi_x

        KE = bm.zeros((12, 12))

        KE[0, 0] = E * AX / LBM
        KE[0, 6] = -KE[0, 0]
        KE[1, 1] = 12 * E * Iz / (1 + FY) / (LBM**3)
        KE[1, 5] = 6 * E * Iz / (1 + FY) / (LBM**2)
        KE[1, 7] = -KE[1, 1]
        KE[1, 11] = KE[1, 5]
        KE[2, 2] = 12 * E * Iy / (1 + FZ) / (LBM**3)
        KE[2, 4] = -6 * E * Iy / (1 + FZ) / (LBM**2)
        KE[2, 8] = -KE[2, 2]
        KE[2, 10] = KE[2, 4]
        KE[3, 3] = G * Ix / LBM
        KE[3, 9] = -KE[3, 3]
        KE[4, 4] = (4 + FZ) * E * Iy / (1 + FZ) / LBM
        KE[4, 8] = 6 * E * Iy / (1 + FZ) / (LBM**2)
        KE[4, 10] = (2 - FZ) * E * Iy / (1 + FZ) / LBM
        KE[5, 5] = (4 + FY) * E * Iz / (1 + FY) / LBM
        KE[5, 7] = -6 * E * Iz / (1 + FY) / (LBM**2)
        KE[5, 11] = (2 - FY) * E * Iz / (1 + FY) / LBM
        KE[6, 6] = KE[0, 0]
        KE[7, 7] = -KE[1, 7]
        KE[7, 11] = -KE[1, 11]
        KE[8, 8] = -KE[2, 8]
        KE[8, 10] = -KE[2, 10]
        KE[9, 9] = KE[3, 3]
        KE[10, 10] = KE[4, 4]
        KE[11, 11] = KE[5, 5]

        # Symmetry of KE matrix
        for j in range(11):
            for i in range(j + 1, 12):
                KE[i, j] = KE[j, i]

        Ke = R.T @ KE @ R  # Matrix multiplication with R

        return Ke