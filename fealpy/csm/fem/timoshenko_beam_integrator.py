from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache)
from fealpy.functionspace.space import FunctionSpace as _FS


class TimoshenkoBeamIntegrator(LinearInt, OpInt, CellInt):
    """
    Integrator for Timoshenko beam problems.
    """

    def __init__(self, 
                 space: _FS, 
                 model,
                 material, 
                 index: Index=_S,
                 method: Optional[str]=None )-> None:
        super().__init__()

        self.space = space
        self.model = model
        self.material = material
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    @enable_cache
    def fetch(self, space: _FS):
        """Retrieve material and geometric parameters for the 3D Timoshenko beam.
        
        Parameters:
            E(float) : Young's modulus.
            mu(float): shear modulus.
            l(float): Length of the beam element.
            AX(float): Cross-sectional area in the x-direction.
            AY(float): Cross-sectional area in the y-direction.
            AZ(float): Cross-sectional area in the z-direction.
            Iy(float): Moment of inertia about the y-axis.
            Iz(float): Moment of inertia about the z-axis.
            Ix(float): Polar moment of inertia (for torsional effects).
        """
        assert space is self.space  
        mesh = space.mesh
        cells = bm.arange(mesh.number_of_cells()) if self.index is _S else self.index
        
        # 参数
        NC = len(cells)
        l = mesh.entity_measure('cell')[cells]
        E, mu = self.material.E, self.material.mu
        Ax, Ay, Az = self.model.Ax, self.model.Ay, self.model.Az
        J, Iy, Iz = self.model.J, self.model.Iy, self.model.Iz
        R = self.model.coord_transform(index=self.index)
        
        return E, mu, l, Ax, Ay, Az, J, Iy, Iz, R, NC

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        """Construct the stiffness matrix for 3D beam elements.
        This function computes the (12, 12) stiffness matrix for each element.
            
        Returns:
            Ke(ndarray),The 3D beam element stiffness matrix, shape (NC, 12, 12).
        """
        assert space is self.space 
        
        E, mu, l, Ax, Ay, Az, J, Iy, Iz, R, NC = self.fetch(space)

        phi_y = 12 * E * Iz / mu / Ay / (l**2)
        phi_z = 12 * E * Iy / mu / Az / (l**2)

        Ke = bm.zeros((NC, 12, 12))

        Ke[:, 0, 0] = E * Ax / l
        Ke[:, 0, 6] = -Ke[:, 0, 0]

        Ke[:, 1, 1] = 12 * E * Iz / (1+phi_y) /(l**3)
        Ke[:, 1, 5] = 6 * E *Iz / (1+phi_y) / (l**2)
        Ke[:, 1, 7] = -Ke[:, 1, 1]
        Ke[:, 1, 11] = Ke[:, 1, 5]

        Ke[:, 2, 2] = 12 * E * Iy / (1+phi_z) /(l**3)
        Ke[:, 2, 4] = -6 * E *Iy / (1+phi_z) / (l**2)
        Ke[:, 2, 8] = -Ke[:, 2, 2]
        Ke[:, 2, 10] = Ke[:, 2, 4]

        Ke[:, 3, 3] = mu * J / l
        Ke[:, 3, 9] = -Ke[:, 3, 3]

        Ke[:, 4, 4] = (4+phi_z) * E * Iy / (1+phi_z) / l
        Ke[:, 4, 8] = 6 * E * Iy / (1+phi_z) / (l**2)
        Ke[:, 4, 10] = (2-phi_z) * E * Iy / (1+phi_z) / l

        Ke[:, 5, 5] = (4+phi_y) * E * Iz / (1+phi_y) / l
        Ke[:, 5, 7] = -6 * E * Iz / (1+phi_y) / (l**2)
        Ke[:, 5, 11] = (2-phi_y) * E * Iz / (1+phi_y) / l

        Ke[:, 6, 6] = Ke[:, 0, 0]
        Ke[:, 7, 7] = -Ke[:, 1, 7]
        Ke[:, 7, 11] = -Ke[:, 1, 11]

        Ke[:, 8, 8] = -Ke[:, 2, 8]
        Ke[:, 8, 10] = -Ke[:, 2, 10]
        Ke[:, 9, 9] = Ke[:, 3, 3]
        Ke[:, 10, 10] = Ke[:, 4, 4]
        Ke[:, 11, 11] = Ke[:, 5, 5]

        # Symmetrize
        for j in range(11):
            for k in range(j + 1, 12):
                Ke[:, k, j] = Ke[:, j, k]

        KE = bm.einsum('cji, cjk, ckl -> cil', R, Ke, R)
        
        return KE