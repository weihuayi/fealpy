from typing import Optional, Literal

from ...backend import backend_manager as bm
from ...typing import TensorLike, Index, _S, SourceLike

from ...functionspace.space import FunctionSpace as _FS
from ...utils import process_coef_func
from ...functional import linear_integral
from ...decorator.variantmethod import variantmethod
from ...fem.integrator import LinearInt, SrcInt, CellInt, enable_cache


class ElastoplasticitySourceIntIntegrator(LinearInt, SrcInt, CellInt):
    """Elastoplasticity Source Internal Integrator
    This class is used for assembling and integrating internal source terms in elastoplastic problems based on function spaces on homogeneous meshes. It is suitable for elastoplastic mechanics simulation in the finite element method, supporting batch processing and multiple integration methods.
    Parameters
        strain_matrix : TensorLike
            Strain matrix describing the strain distribution within the element.
        stress : SourceLike, optional, default=None
            Stress tensor or source term, default is None.
        q : int, optional, default=None
            Integration order, if None it is set automatically according to the space order.
        region : TensorLike, optional, default=None
            Specifies the integration region, default is None for the whole domain.
        batched : bool, optional, default=False
            Whether to use batch processing mode.
        method : Literal['isopara', None], optional, default=None
            Integration method selection, supports 'isopara' or None.
    Attributes
        strain_matrix : TensorLike
            Strain matrix within the element.
        stress : SourceLike or None
            Stress tensor or source term within the element.
        q : int or None
            Integration order.
        batched : bool
            Whether batch processing is used.
        region : TensorLike or None
            Integration region.
    Methods
        to_global_dof(space, indices=None)
            Get global degree of freedom numbers.
        fetch(space, inidces=None)
            Get basis functions, weights, measures, and other information required for integration.
        assembly(space, indices=None)
            Assemble the internal source term for elastoplasticity.
    Notes
        This class combines finite element space and mesh information to efficiently integrate and assemble internal source terms for elastoplasticity. It supports caching mechanisms to improve performance and allows flexible selection of integration regions and methods.
    Examples
        >>> integrator = ElastoplasticitySourceIntIntegrator(strain_matrix, stress, q=3)
        >>> F_int = integrator.assembly(space)
    """
    """The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, strain_matrix, stress: Optional[SourceLike]=None, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False,
                 method: Literal['isopara', None] = None) -> None:
        super().__init__()
        self.strain_matrix = strain_matrix
        self.stress = stress
        self.q = q
        self.set_region(region)
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        if indices is None:
            return space.cell_to_dof()
        return space.cell_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch(self, space: _FS, /, inidces=None):
        q = self.q
        index = self.entity_selection(inidces)
        mesh = getattr(space, 'mesh', None)
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)

        return bcs, ws, phi, cm, index

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space, indices)
        F_int = bm.einsum('q, c, cqij,cqi->cj', 
                                ws, cm, self.strain_matrix, self.stress) # (NC, tdof)

        return F_int