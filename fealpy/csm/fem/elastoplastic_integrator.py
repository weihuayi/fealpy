#from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S

from fealpy.mesh import HomogeneousMesh, SimplexMesh, TensorMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functionspace.tensor_space import TensorFunctionSpace as _TS
from fealpy.fem.integrator import (
    LinearInt, OpInt, CellInt
)
from fealpy.fem import LinearElasticityIntegrator
from fealpy.fem.linear_form import LinearForm

class ElastoplasticIntegrator(LinearElasticityIntegrator):
    '''
    ElastoplasticIntegrator integrates the constitutive behavior of elastoplastic materials within a finite element framework.
    This class extends the LinearElasticityIntegrator to handle elastoplastic constitutive updates, internal force computation, and tangent stiffness matrix assembly. It is designed for use in computational solid mechanics simulations where both elastic and plastic material responses are present.
    Parameters
    D_ep : array-like
        Elastoplastic material stiffness matrix, typically of shape (n_cells, n_qp, n_strain, n_strain).
    space : FunctionSpace
        The finite element function space used for the discretization.
    material : Material
        Material model object providing elastic and plastic constitutive behavior.
    q : int
        Quadrature order for numerical integration.
    equivalent_plastic_strain : array-like
        Array storing the equivalent plastic strain at integration points.
    method : str or None, optional, default=None
        Integration method or scheme to be used.
    Attributes
    D_ep : array-like
        Elastoplastic material stiffness matrix used in tangent computations.
    space : FunctionSpace
        The finite element function space associated with the integrator.
    equivalent_plastic_strain : array-like
        Stores the equivalent plastic strain at each integration point.
    Methods
    compute_internal_force(uh, plastic_strain, index=_FS)
        Compute the internal force vector considering plastic strain effects.
    constitutive_update(uh, plastic_strain_old, material, yield_stress, strain_total_e)
        Perform constitutive integration and return updated state variables.
    update_elastoplastic_matrix(material, n, yield_mask)
        Construct the consistent elastoplastic tangent matrix.
    assembly(space)
        Assemble the global tangent stiffness matrix for the current state.
    Notes
    This class assumes small strain elastoplasticity and is suitable for incremental-iterative solution procedures such as Newton-Raphson. The implementation supports von Mises plasticity and can be extended for other yield criteria.
    Examples
    >>> integrator = ElasticoplasticIntegrator(D_ep, space, material, q)
    >>> F_int = integrator.compute_internal_force(uh, plastic_strain)
    >>> converged, plastic_strain_new, D_ep_new, strain_total_e = integrator.constitutive_update(
    ...     uh, plastic_strain_old, material, yield_stress, strain_total_e)
    >>> K_tangent = integrator.assembly(space)
    '''
    def __init__(self, D_ep, space, material, q, method=None):
        # 传递 method 参数并调用父类构造函数
        super().__init__(material, q, method=method)
        self.D_ep = D_ep  # 弹塑性材料矩阵
        self.space = space  # 函数空间

    def compute_internal_force(self, uh, plastic_strain,index=_FS) -> TensorLike:
        """计算考虑塑性应变的内部力"""
        space = self.space
        mesh = space.mesh
        node = mesh.entity('node')
        kwargs = bm.context(node)

        # 获取单元局部位移
        cell2dof = space.cell_to_dof()
        uh = bm.array(uh,**kwargs)  
        tldof = space.number_of_local_dofs()
        uh_cell = uh[cell2dof]
        qf = mesh.quadrature_formula(q=space.p+3)   
        bcs, ws = qf.get_quadrature_points_and_weights()
        # 计算应变
        B = self.material.strain_matrix(True, gphi=space.scalar_space.grad_basis(bcs))
        D = self.material.elastic_matrix()
        strain_total = bm.einsum('cqij,cj->cqi', B, uh_cell)
        strain_elastic = strain_total - plastic_strain

        # 计算应力
        stress = bm.einsum('cqij,cqj->cqi', D, strain_elastic)

        # 组装内部力
        cm = mesh.entity_measure('cell')
        F_int_cell = bm.einsum('q, c, cqij,cqi->cj', 
                             ws, cm, B, stress) # (NC, tdof)
        
        return F_int_cell     

    def assembly(self, space: _FS) -> TensorLike:
        '''组装切线刚度矩阵'''
        mesh = getattr(space, 'mesh', None)
        D_ep = self.D_ep
        cm, ws, detJ, D, B = self.fetch_voigt_assembly(space)
        
        if isinstance(mesh, TensorMesh):
            KK = bm.einsum('c, cq, cqki, cqkl, cqlj -> cij',
                            ws, detJ, B, D_ep, B)
        else:
            KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij',
                            ws, cm, B, D_ep, B)
        
        return KK # (NC, tdof, tdof)