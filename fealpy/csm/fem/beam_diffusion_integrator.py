from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
)
from fealpy.functionspace.tensor_space import TensorFunctionSpace as _TS
from typing import Optional, Literal
from fealpy.typing import TensorLike, Index, _S
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.decorator import variantmethod
class BeamDiffusionIntegrator(LinearInt, OpInt, CellInt):
    """
    Short Description
    BeamDiffusionIntegrator assembles and manages the diffusion (stiffness) matrix for beam elements in a finite element space.

    Detailed Description
    This class supports different types of beam elements ('euler_bernoulli_2d', 'normal_2d', 'euler_bernoulli_3d') and precomputes relevant stiffness coefficients for efficient finite element analysis. It is designed for structural mechanics applications involving beam modeling and analysis.

    Parameters
    space : object
        The finite element space object defining element DOFs and interpolation functions.
    beam_type : str
        Type of beam: 'euler_bernoulli_2d' (pure bending), 'normal_2d' (2D beam), or 'euler_bernoulli_3d' (3D beam).
    E : float
        Young's modulus of the material.
    l : array_like
        Array of beam element lengths, shape (NC,).
    A : float, optional, default=None
        Cross-sectional area (required for normal_2d/euler_bernoulli_3d beams).
    I : float, optional, default=None
        Moment of inertia (required for euler_bernoulli_2d/normal_2d beams).
    Iy : float, optional, default=None
        Moment of inertia about y-axis (required for euler_bernoulli_3d beams).
    Iz : float, optional, default=None
        Moment of inertia about z-axis (required for euler_bernoulli_3d beams).
    G : float, optional, default=None
        Shear modulus (required for euler_bernoulli_3d beams).
    J : float, optional, default=None
        Torsional constant (required for euler_bernoulli_3d beams).
    method : str, optional, default='assembly'
        Assembly method, default is 'assembly'.

    Attributes
    space : object
        The finite element space object.
    type : str
        Beam type: 'euler_bernoulli_2d', 'normal_2d', or 'euler_bernoulli_3d'.
    E : float
        Young's modulus.
    l : array_like
        Array of beam element lengths.
    A : float or None
        Cross-sectional area.
    I : float or None
        Moment of inertia.
    EA : float or None
        Axial stiffness (for normal_2d/euler_bernoulli_3d beams).
    EI : float or None
        Bending stiffness (for normal_2d beams).
    EIy : float or None
        Bending stiffness about y-axis (for euler_bernoulli_3d beams).
    EIz : float or None
        Bending stiffness about z-axis (for euler_bernoulli_3d/euler_bernoulli_2d beams).
    GJ : float or None
        Torsional stiffness (for euler_bernoulli_3d beams).

    Methods
    to_global_dof(space)
        Maps local DOFs to global DOFs.
    assembly(space)
        Assembles the element stiffness matrix for the specified beam type.

    Notes
    Required parameters depend on the beam type. The class automatically checks and computes necessary stiffness coefficients during initialization.

    Examples
    >>> integrator = BeamDiffusionIntegrator(space, 'normal_2d', E=210e9, l=[1.0, 1.0], A=0.01, I=1e-6)
    >>> print(integrator.EA)
    >>> print(integrator.EI)
    """
    def __init__(self, 
                 space, beam_type, 
                 E, l, A=None, I=None, Iy=None, Iz=None, G=None, J=None,
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
        self.type = beam_type.lower()
        self.E, self.l = E, l
        self.I = I
        self.A = A
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        if self.type == 'euler_bernoulli_2d':
            assert I is not None
            self.EIz = E * I
        elif self.type == 'normal_2d':
            assert A is not None and I is not None
            self.EA = E * A
            self.EI = E * I
        elif self.type == 'euler_bernoulli_3d':
            assert all(v is not None for v in [A, Iy, Iz, G, J])
            self.EA = E * A
            self.EIy = E * Iy
            self.EIz = E * Iz
            self.GJ = G * J
        else:
            raise ValueError(f"未知梁类型: {self.type}")
        
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()

    @variantmethod("euler_bernoulli_2d")
    def assembly(self, space: _TS) -> TensorLike:
        """
        Construct the stiffness matrix for pure bending beam elements.
        This function computes the (4, 4) pure bending stiffness matrix for each element
        based on the physical parameters (Young's modulus E, moment of inertia I, element length l).
        It is suitable for 2D Euler-Bernoulli beam elements and efficiently assembles the stiffness
        matrices for all elements using block matrix operations.

        Parameters
        self : object
            The object containing beam element parameters, with attributes E (Young's modulus),
            I (moment of inertia), and l (element length), all as 1D arrays of shape (NC,), where NC is the number of elements.

        Returns
        Ke : ndarray
            The pure bending beam element stiffness matrix, shape (NC, 4, 4), with each element corresponding to a (4, 4) matrix.

        Raises
        AssertionError
            Raised if self.I is None, indicating that moment of inertia I must be provided for pure bending beams.

        Notes
        The stiffness matrix is derived from Euler-Bernoulli beam theory and is suitable for small deformation linear elastic analysis.
        """
        assert self.I is not None, "纯弯梁需要提供 I"
        EIz = self.E * self.I       # (NC,)
        l = self.l                  # (NC,)
        coef = EIz / l**3           # (NC,)

      
        k00 = 12 * coef
        k01 = 6 * l * coef
        k02 = -12 * coef
        k03 = 6 * l * coef

        k11 = 4 * l**2 * coef
        k12 = -6 * l * coef
        k13 = 2 * l**2 * coef

        k22 = 12 * coef
        k23 = -6 * l * coef

        k33 = 4 * l**2 * coef


        Ke = bm.stack([
            bm.stack([k00, k01, k02, k03], axis=-1),
            bm.stack([k01, k11, k12, k13], axis=-1),
            bm.stack([k02, k12, k22, k23], axis=-1),
            bm.stack([k03, k13, k23, k33], axis=-1),
        ], axis=1)  

        return Ke

    @assembly.register("normal_2d")
    def assembly(self, space: _TS) -> TensorLike:
        """
        Construct the stiffness matrix for 2D beam elements.
        This function computes the (6, 6) stiffness matrix for each element
        based on the physical parameters (Young's modulus E, cross-sectional area A,
        moment of inertia I, element length l). It is suitable for 2D Euler-Bernoulli
        beam elements and efficiently assembles the stiffness matrices for all elements
        using block matrix operations.
        Parameters
        self : object
            The object containing beam element parameters, with attributes E (Young's modulus),
            A (cross-sectional area), I (moment of inertia), and l (element length), all as 1D arrays of shape (NC,), where NC is the number of elements.
        Returns
        Ke : ndarray
            The 2D beam element stiffness matrix, shape (NC, 6, 6), with each element corresponding to a (6, 6) matrix.
        Raises
        AssertionError
            Raised if self.A or self.I is None, indicating that cross-sectional area A and moment of inertia I must be provided for 2D beams.
        Notes
        The stiffness matrix is derived from Euler-Bernoulli beam theory and is suitable for small deformation linear elastic analysis.
        """
        assert self.A is not None and self.I is not None, "二维梁需要提供 A 和 I"
        l = self.l[:, None, None]
        EA_l = self.EA / l
        EI_l3 = self.EI / l**3
        EI_l2 = self.EI / l**2
        EI_l = self.EI / l

        Ke = bm.array([
            [ EA_l,    0,        0,       -EA_l,   0,        0],
            [ 0,     12*EI_l3, 6*EI_l2,   0,    -12*EI_l3, 6*EI_l2],
            [ 0,     6*EI_l2,  4*EI_l,    0,     -6*EI_l2, 2*EI_l],
            [-EA_l,   0,        0,        EA_l,   0,        0],
            [ 0,    -12*EI_l3,-6*EI_l2,   0,     12*EI_l3,-6*EI_l2],
            [ 0,     6*EI_l2,  2*EI_l,    0,     -6*EI_l2, 4*EI_l],
        ])
        return Ke
    
    @assembly.register("euler_bernoulli_3d")
    def assembly(self, space: _TS) -> TensorLike:
        """
        Construct the stiffness matrix for 3D beam elements.
        This function computes the (12, 12) stiffness matrix for each element
        based on the physical parameters (Young's modulus E, cross-sectional area A,
        moments of inertia Iy and Iz, shear modulus G, torsional constant J,
        element length l). It is suitable for 3D Euler-Bernoulli beam elements
        and efficiently assembles the stiffness matrices for all elements
        using block matrix operations.
        Parameters
        self : object
            The object containing beam element parameters, with attributes E (Young's modulus),
            A (cross-sectional area), Iy (moment of inertia about y-axis), Iz (moment of inertia about z-axis),
            G (shear modulus), J (torsional constant), and l (element length), all as 1D arrays of shape (NC,), where NC is the number of elements.
        Returns
        Ke : ndarray
            The 3D beam element stiffness matrix, shape (NC, 12, 12), with each element corresponding to a (12, 12) matrix.
        Raises
        AssertionError
            Raised if self.A, self.Iy, self.Iz, self.G, or self.J is None, indicating that cross-sectional area A,
            moments of inertia Iy and Iz, shear modulus G, and torsional constant J must be provided for 3D beams.
        Notes
        The stiffness matrix is derived from Euler-Bernoulli beam theory and is suitable for small deformation linear elastic analysis.
        """
        assert all(v is not None for v in [self.A, self.Iy, self.Iz, self.G, self.J]), "三维梁需要提供 A, Iy, Iz, G, J"
        l = self.l[:, None, None]
        NC = l.shape[0]

        EA_l  = self.EA / l
        EIy_l3 = self.EIy / l**3
        EIy_l2 = self.EIy / l**2
        EIy_l  = self.EIy / l
        EIz_l3 = self.EIz / l**3
        EIz_l2 = self.EIz / l**2
        EIz_l  = self.EIz / l
        GJ_l = self.GJ / l

        Ke = bm.zeros((NC, 12, 12))

        # 轴向
        Ke[:,0,0]  = EA_l[...,0,0]
        Ke[:,0,6]  = -EA_l[...,0,0]
        Ke[:,6,0]  = -EA_l[...,0,0]
        Ke[:,6,6]  = EA_l[...,0,0]

        # 绕Z弯曲（Y方向位移）
        Ke[:,1,1]   = 12*EIz_l3[...,0,0]
        Ke[:,1,5]   = 6*EIz_l2[...,0,0]
        Ke[:,1,7]   = -12*EIz_l3[...,0,0]
        Ke[:,1,11]  = 6*EIz_l2[...,0,0]
        Ke[:,5,1]   = 6*EIz_l2[...,0,0]
        Ke[:,5,5]   = 4*EIz_l[...,0,0]
        Ke[:,5,7]   = -6*EIz_l2[...,0,0]
        Ke[:,5,11]  = 2*EIz_l[...,0,0]
        Ke[:,7,1]   = -12*EIz_l3[...,0,0]
        Ke[:,7,5]   = -6*EIz_l2[...,0,0]
        Ke[:,7,7]   = 12*EIz_l3[...,0,0]
        Ke[:,7,11]  = -6*EIz_l2[...,0,0]
        Ke[:,11,1]  = 6*EIz_l2[...,0,0]
        Ke[:,11,5]  = 2*EIz_l[...,0,0]
        Ke[:,11,7]  = -6*EIz_l2[...,0,0]
        Ke[:,11,11] = 4*EIz_l[...,0,0]

        # 绕Y弯曲（Z方向位移）
        Ke[:,2,2]   = 12*EIy_l3[...,0,0]
        Ke[:,2,4]   = -6*EIy_l2[...,0,0]
        Ke[:,2,8]   = -12*EIy_l3[...,0,0]
        Ke[:,2,10]  = -6*EIy_l2[...,0,0]
        Ke[:,4,2]   = -6*EIy_l2[...,0,0]
        Ke[:,4,4]   = 4*EIy_l[...,0,0]
        Ke[:,4,8]   = 6*EIy_l2[...,0,0]
        Ke[:,4,10]  = 2*EIy_l[...,0,0]
        Ke[:,8,2]   = -12*EIy_l3[...,0,0]
        Ke[:,8,4]   = 6*EIy_l2[...,0,0]
        Ke[:,8,8]   = 12*EIy_l3[...,0,0]
        Ke[:,8,10]  = 6*EIy_l2[...,0,0]
        Ke[:,10,2]  = -6*EIy_l2[...,0,0]
        Ke[:,10,4]  = 2*EIy_l[...,0,0]
        Ke[:,10,8]  = 6*EIy_l2[...,0,0]
        Ke[:,10,10] = 4*EIy_l[...,0,0]

        # 扭转（绕X）
        Ke[:,3,3] = GJ_l[...,0,0]
        Ke[:,3,9] = -GJ_l[...,0,0]
        Ke[:,9,3] = -GJ_l[...,0,0]
        Ke[:,9,9] = GJ_l[...,0,0]

        return Ke