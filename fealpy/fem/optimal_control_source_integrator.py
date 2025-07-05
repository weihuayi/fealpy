from typing import Optional
from functools import partial

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func, is_scalar, is_tensor, fill_axis
from fealpy.functional import bilinear_integral, linear_integral, get_semilinear_coef
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
)


class OPCSIntegrator(LinearInt, OpInt, CellInt):
    """
    OPCSIntegrator: Optimal Control Source Integrator

    Assembles the source term for optimal control problems on 2D triangular meshes,
    suitable for finite element methods such as Raviart-Thomas elements.
    Supports custom source functions, selectable quadrature order, and achieves
    high-accuracy source assembly via integration over sub-triangles.

    Parameters
    source : CoefLike, optional
        The source function or tensor for the optimal control problem.
    q : int, optional
        Quadrature order. If None, it is automatically chosen based on the polynomial degree.
    index : Index, optional
        Indices of the cells to assemble. Defaults to all cells.
    batched : bool, optional
        Whether to use batched assembly for performance.
    method : str, optional
        Assembly method, default is 'assembly'.

    Methods
    to_global_dof(space)
        Maps local cell degrees of freedom to global degrees of freedom.
    transform_subtriangle_points(quad_points)
        Transforms barycentric quadrature points of sub-triangles to global coordinates of the original triangle.
    fetch(space)
        Generates quadrature points, weights, basis functions, and cell measures for assembly.
    assembly(space)
        Assembles the global source vector for the optimal control problem.

    Notes
    This integrator is suitable for high-accuracy source integration in optimal control PDE finite element solvers,
    especially for mixed elements such as Raviart-Thomas.
    """
    def __init__(self, source: Optional[TensorLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.source = source
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]
    
    def transform_subtriangle_points(self, quad_points):
        '''
        Transform quadrature points from subtriangles to the global coordinates of the original triangle.
        This function maps the quadrature points defined in the barycentric coordinates of each subtriangle
        (obtained by subdividing the original triangle) to the barycentric coordinates of the original triangle.
        It applies a specific affine transformation for each subtriangle and returns the transformed points.
        Parameters
        quad_points : ndarray
            Quadrature points in the barycentric coordinates of the subtriangle, shape (NQ, 3).
        Returns
        global_quad_points : ndarray
            Quadrature points in the barycentric coordinates of the original triangle, shape (3*NQ, 3).
        subtri1 : ndarray
            Transformed quadrature points for the first subtriangle, shape (NQ, 3).
        subtri2 : ndarray
            Transformed quadrature points for the second subtriangle, shape (NQ, 3).
        subtri3 : ndarray
            Transformed quadrature points for the third subtriangle, shape (NQ, 3).
        Notes
        The transformation is performed using predefined matrices for each subtriangle, corresponding to
        the subdivision of the original triangle along its edges.
        Examples
        >>> quad_points = np.array([[1/3, 1/3, 1/3]])
        >>> global_quad_points, subtri1, subtri2, subtri3 = transform_subtriangle_points(quad_points)
        >>> print(global_quad_points.shape)
        (3, 3)
        '''
        # 定义变换矩阵
        # 第0条边
        M0 = bm.array([
            [0, 0, 1/3],
            [1, 0, 1/3],
            [0, 1, 1/3]
        ])
    
        
        # 第1条边
        M1 = bm.array([
            [0, 1, 1/3],
            [0, 0, 1/3],
            [1, 0, 1/3]
        ])
        # 第2条边
        M2 = bm.array([
            [1, 0, 1/3],
            [0, 1, 1/3],
            [0, 0, 1/3]
        ])


        # 对每个小三角形应用变换
        subtri1 = bm.dot(quad_points, M0.T)  # 第一个小三角形
        subtri2 = bm.dot(quad_points, M1.T)  # 第二个小三角形
        subtri3 = bm.dot(quad_points, M2.T)  # 第三个小三角形

        # 合并所有积分点
        global_quad_points = bm.stack([subtri1, subtri2, subtri3])
        return global_quad_points
    
    @enable_cache
    def fetch(self, space: _FS):
        '''
        Fetch the necessary data for assembly, including quadrature points, weights,
        basis functions, and cell measures.
        Parameters
        space : _FS
            The function space on which the integrator operates.
        Returns
        bcs : TensorLike
            Barycentric coordinates of the quadrature points.
        global_points : TensorLike     
            Global coordinates of the quadrature points transformed from subtriangles.
        ws : TensorLike
            Weights of the quadrature points.
        phi_dual : TensorLike   
            Basis functions evaluated at the quadrature points.
        cm : TensorLike 
            Cell measures for the integration.
        index : Index   
            Indices of the cells to assemble.
        Notes   
        This method retrieves the quadrature points, weights, and basis functions
        for the given function space, specifically designed for optimal control problems
        on homogeneous meshes. It supports high-order quadrature and transforms
        barycentric coordinates of sub-triangles to the global coordinates of the original triangle.
        Raises
        RuntimeError
        If the mesh is not homogeneous, as this integrator is designed for homogeneous meshes only.
        Examples
        >>> integrator = OPCSIntegrator(source=my_source, q=4)
        >>> bcs, global_points, ws, phi_dual, cm, index = integrator.fetch(my_function_space)
        '''
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        global_points =self.transform_subtriangle_points(bcs)
        bcs_dual = bm.array([[0, 0.5, 0.5],[0.5,0, 0.5],[0.5,0.5,0]])
        phi_dual = space.basis(bcs_dual, index=index)
        #phi_new_1: 第0个基函数  
        phi_new_1 = phi_dual[:, 0, 0, :]  
        # phi_new_2: 第1个基函数
        phi_new_2 = phi_dual[:, 1, 1, :]  
        # phi_new_3: 第2个基函数
        phi_new_3 = phi_dual[:, 2, 2, :]  
        # 将 phi_new_1, phi_new_2, phi_new_3 堆叠起来
        phi_dual = bm.stack([phi_new_1, phi_new_2, phi_new_3], axis=1) 
        phi_dual = phi_dual[:,bm.newaxis,:,:] 
        return bcs, global_points, ws, 2*phi_dual, cm, index
    
    
    def assembly(self,  space: _FS) -> TensorLike:
        '''
        Assemble the global source vector for the optimal control problem.
        Parameters
        space : _FS
            The function space on which the integrator operates.
        Returns
        result : TensorLike
            The assembled source vector, shape (number_of_global_dofs,).
        Notes
        This method computes the source term for optimal control problems by integrating
        the source function over the mesh using the quadrature points and weights.
        It supports both scalar and tensor source functions, and handles the assembly
        using the dual basis functions for Raviart-Thomas elements.
        If the source function is a scalar, it computes the integral over the mesh
        using the dual basis functions and the cell measures. If the source function
        is a tensor, it computes the integral using the dual basis functions and
        the cell measures, applying the appropriate tensor operations.
        '''
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, global_points, ws, phi_dual, cm, index = self.fetch(space)
        val0 = process_coef_func(f, bcs=global_points[0], mesh=mesh, etype='cell', index=index)
        val1 = process_coef_func(f, bcs=global_points[1], mesh=mesh, etype='cell', index=index)
        val2 = process_coef_func(f, bcs=global_points[2], mesh=mesh, etype='cell', index=index)
        val = bm.stack([val0, val1, val2])
        if isinstance(f, int):
            result = bm.einsum('q,c,i,cqik->ci', ws, 1/3*cm, val, phi_dual)
        else:
            result = bm.einsum('q, c, icqk, cqik -> ci', ws, 1/3*cm, val, phi_dual)
        return result