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


class OPCIntegrator(LinearInt, OpInt, CellInt):

    def __init__(self, coef: Optional[TensorLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        """
        OPCIntegrator: Optimal Control Problem Cell Integrator
        This class implements a cell integrator for optimal control problems in the finite element method (FEM).
        It is designed to assemble local matrices or vectors on mesh cells, supporting custom coefficients,
        quadrature order, and batched processing for efficient numerical integration.

        Parameters
        coef : Optional[CoefLike], optional
            Coefficient(s) used in the integrand, by default None.
        q : Optional[int], optional
            Quadrature order for numerical integration, by default None.
        index : Index, optional, default=_S
            Index specifying the structure or layout of the data.
        batched : bool, optional, default=False
            Whether to enable batched processing for performance optimization.
        method : Optional[str], optional
            Method of operation, defaults to 'assembly' if not provided.

        Attributes
        coef : Optional[CoefLike]
            Coefficient(s) used in the computation.
        q : Optional[int]
            Quadrature order for numerical integration.
        index : Index
            Index specifying the structure or layout of the data.
        batched : bool
            Indicates if batched processing is enabled.
        method : str
            Method of operation, either provided or defaulted to 'assembly'.

        Notes
        The `method` parameter defaults to 'assembly' if not explicitly specified.
        This class is intended for use in optimal control FEM assembly routines,
        supporting flexible integration strategies and efficient batched computation.
        """
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
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
        """
        Fetch the necessary data for the assembly operation.
        This method retrieves the quadrature points, weights, and basis functions
        for the given function space.
        Parameters
        space : _FS
            The function space on which to perform the assembly operation.
        Returns
        bcs : TensorLike
            The quadrature points in the reference element.
        global_points : TensorLike
            The quadrature points in the global coordinate system.
        ws : TensorLike
            The weights associated with the quadrature points.
        phi : TensorLike
            The basis functions evaluated at the quadrature points.
        phi_dual : TensorLike
            The dual basis functions evaluated at the quadrature points.
        cm : TensorLike
            The cell measure associated with the quadrature points.
        index : Index
            The index of the function space.
        """
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
        global_points = self.transform_subtriangle_points(bcs)
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
        phi = space.basis(bcs, index=index)
        phi_M1 = space.basis(global_points[0], index=index)
        phi_M2 = space.basis(global_points[1], index=index)
        phi_M3 = space.basis(global_points[2], index=index)
        phi = bm.stack([phi_M1, phi_M2, phi_M3])
        return global_points, ws, phi, 2*phi_dual, cm, index


    def assembly(self, space: _FS) -> TensorLike:
        """
        Perform the assembly operation for the given function space.

        This method computes the integral using the specified coefficients and
        quadrature points, and returns the result as a tensor.

        Parameters
        space : _FS
            The function space on which to perform the assembly.

        Returns
        TensorLike
            The result of the assembly operation.
        """
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        global_points, ws, phi, phi_dual, cm, index = self.fetch(space)
        
        NC = phi.shape[1]
        ldof = phi.shape[3]
        result = bm.zeros((NC, ldof, ldof), dtype=phi.dtype)
        phi_dua11 = phi_dual[:,:,0,:]
        phi_dua12 = phi_dual[:,:,1,:]
        phi_dua13 = phi_dual[:,:,2,:]
        phi1 = phi[0,...]
        phi2 = phi[1,...]
        phi3 = phi[2,...]
        val1 = process_coef_func(coef, bcs=global_points[0], mesh=mesh, etype='cell', index=index)
        val2 = process_coef_func(coef, bcs=global_points[1], mesh=mesh, etype='cell', index=index)
        val3 = process_coef_func(coef, bcs=global_points[2], mesh=mesh, etype='cell', index=index)

        phi1_val = bm.einsum('cqlk, cqdk -> cqld',  phi1, val1 ) #TODO：check the einsum
        phi2_val = bm.einsum('cqlk, cqdk -> cqld',  phi2, val2 )
        phi3_val = bm.einsum('cqlk, cqdk -> cqld',  phi3, val3 )
        result1 = bm.einsum('q, c, cqlk, cqk -> cl', ws, 1/3*cm, phi1_val, phi_dua11)
        result2 = bm.einsum('q, c, cqlk, cqk -> cl', ws, 1/3*cm, phi2_val, phi_dua12)
        result3 = bm.einsum('q, c, cqlk, cqk -> cl', ws, 1/3*cm, phi3_val, phi_dua13)
        result[:,0,:] = result1
        result[:,1,:] = result2 
        result[:,2,:] = result3
        #result = result.transpose(0, 2, 1)
        
        return result
    