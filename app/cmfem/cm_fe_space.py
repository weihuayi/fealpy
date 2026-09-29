from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.functionspace import BernsteinFESpace
from fealpy.functionspace.space import FunctionSpace
from fealpy.sparse import COOTensor

from simplex_lattice import SimplexLattice
from sparse_tensor_ops import spmv, spsolve_triangular
from symmetric_tensor import symmetry_multiplicity
from bernstein_assembly import smooth_stiffness


class CmFESpace(FunctionSpace):
    def __init__(self, mesh, p, r, dof):
        self.mesh = mesh
        self.p = p
        self.r = tuple(r)
        self.ftype = mesh.ftype
        self.itype = bm.int32
        self.device = mesh.device
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()
        self.lattice = SimplexLattice(self.TD, p, self.r)
        self.dof = dof(mesh, self.lattice)
        self.bspace = BernsteinFESpace(mesh, p=p, ctype='D')
        self._coeff = None

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_internal_dofs(self, etype):
        return self.dof.number_of_internal_dofs(etype)

    def cell_to_dof(self, index=slice(None)):
        return self.dof.cell_to_dof()[index]

    def coefficient_matrix(self):
        """Return the Bernstein coefficient matrix ``C=T^T D^{-T}``."""
        T = self.T.to_dense()
        CT = spsolve_triangular(self.D, T, lower=True, unit_diagonal=True)
        return bm.swapaxes(CT, -1, -2)

    @property
    def coeff(self):
        if self._coeff is None:
            self._coeff = self.coefficient_matrix()
        return self._coeff

    @barycentric
    def basis(self, bcs, index=slice(None)):
        phi = self.bspace.basis(bcs)[0]
        phi = phi[:, self.lattice.permutation]
        return bm.einsum('cil,ql->cqi', self.coeff[index], phi)

    @barycentric
    def grad_m_basis(self, bcs, m, index=slice(None)):
        if m == 0:
            return self.basis(bcs, index=index)
        phi = self.bspace.grad_m_basis(bcs, m, index=index)
        phi = phi[:, :, self.lattice.permutation, :]
        return bm.einsum('cil,cqlg->cqig', self.coeff[index], phi)

    @barycentric
    def grad_basis(self, bcs, index=slice(None)):
        return self.grad_m_basis(bcs, 1, index=index)

    @barycentric
    def value(self, uh, bcs, index=slice(None)):
        phi = self.basis(bcs, index=index)
        dof = self.cell_to_dof(index=index)
        return bm.einsum('cqi,...ci->...cq', phi, uh[..., dof])

    @barycentric
    def grad_m_value(self, uh, bcs, m, index=slice(None)):
        if m == 0:
            return self.value(uh, bcs, index=index)
        phi = self.grad_m_basis(bcs, m, index=index)
        dof = self.cell_to_dof(index=index)
        return bm.einsum('cqig,...ci->...cqg', phi, uh[..., dof])

    @barycentric
    def grad_value(self, uh, bcs, index=slice(None)):
        return self.grad_m_value(uh, bcs, 1, index=index)

    def stiffness_matrix(self, order=None, q=None, batch_size=None,
                         method='bernstein'):
        if order is None:
            order = self.r[-2]+1
        if method == 'bernstein':
            local = smooth_stiffness(self, order, q)
        elif method == 'quadrature':
            local = self._quadrature_stiffness(order, q, batch_size)
        else:
            raise ValueError("method must be 'bernstein' or 'quadrature'.")

        dof = self.cell_to_dof()
        shape = local.shape
        row = bm.broadcast_to(dof[:, :, None], shape)
        col = bm.broadcast_to(dof[:, None, :], shape)
        indices = bm.stack((row.reshape(-1), col.reshape(-1)), axis=0)
        size = self.number_of_global_dofs()
        matrix = COOTensor(indices, local.reshape(-1), (size, size))
        return matrix.coalesce().tocsr()

    def _quadrature_stiffness(self, order, q=None, batch_size=None):
        if q is None:
            q = self.p+1

        quadrature = self.mesh.quadrature_formula(q, 'cell')
        bcs, weights = quadrature.get_quadrature_points_and_weights()
        dim = self.mesh.top_dimension()
        mult = symmetry_multiplicity(order, dim, dtype=self.ftype)
        measure = self.mesh.entity_measure('cell')
        nc = self.mesh.number_of_cells()
        if batch_size is None:
            batch_size = nc

        matrices = []
        for start in range(0, nc, batch_size):
            index = slice(start, min(start+batch_size, nc))
            phi = self.grad_m_basis(bcs, order, index=index)
            local = bm.einsum('cqig,cqjg,g,q,c->cij', phi, phi, mult,
                              weights, measure[index])
            matrices.append(local)
        return bm.concatenate(matrices, axis=0)

    def source_vector(self, source, q=None, batch_size=None):
        if q is None:
            q = self.p+1

        quadrature = self.mesh.quadrature_formula(q, 'cell')
        bcs, weights = quadrature.get_quadrature_points_and_weights()
        measure = self.mesh.entity_measure('cell')
        vector = bm.zeros(self.number_of_global_dofs(), dtype=self.ftype)
        nc = self.mesh.number_of_cells()
        if batch_size is None:
            batch_size = nc

        for start in range(0, nc, batch_size):
            index = slice(start, min(start+batch_size, nc))
            phi = self.basis(bcs, index=index)
            point = self.mesh.bc_to_point(bcs, index=index)
            value = source(point)
            local = bm.einsum('cq,cqi,q,c->ci', value, phi, weights,
                              measure[index])
            dof = self.cell_to_dof(index=index)
            vector = bm.index_add(vector, dof.reshape(-1), local.reshape(-1),
                                  axis=0)
        return vector

    def _cell_interpolation(self, derivatives):
        vertices = tuple(range(self.TD+1))
        rows = self.lattice.indices(vertices)
        bcs = self.lattice.multi_index/self.p
        point = self.mesh.bc_to_point(bcs)
        value = derivatives[0](point)
        matrix = self.bspace.basis(bcs)[0]
        right = bm.swapaxes(value, 0, 1)
        coefficient = bm.linalg.solve(matrix, right)
        coefficient = bm.swapaxes(coefficient, 0, 1)
        return coefficient[:, rows]

    def error_norms(self, uh, exact, order, q=None, batch_size=None,
                    frobenius=True):
        """Compute error norms from derivative order zero through ``order``."""
        if q is None:
            q = self.p+4

        quadrature = self.mesh.quadrature_formula(q, 'cell')
        bcs, weights = quadrature.get_quadrature_points_and_weights()
        measure = self.mesh.entity_measure('cell')
        permutation = self.lattice.permutation

        local_value = uh[self.cell_to_dof()]
        local_value = spmv(self.T, local_value)
        coefficient = spsolve_triangular(self.D, local_value,
                                         lower=True, unit_diagonal=True)

        basis = self.bspace.basis(bcs)[0][:, permutation]
        derivatives = []
        dimension = self.mesh.top_dimension()
        for s in range(1, order+1):
            derivative, geometry = self.bspace.grad_m_basis(bcs, s,
                                                             variable='lambda')
            derivative = derivative[:, :, permutation]
            if frobenius:
                multiplicity = symmetry_multiplicity(s, dimension,
                                                     dtype=self.ftype)
            else:
                multiplicity = bm.ones(geometry.shape[-1], dtype=self.ftype)
            data = derivative, geometry, multiplicity
            derivatives.append(data)

        nc = self.mesh.number_of_cells()
        if batch_size is None:
            batch_size = nc
        errors = bm.zeros(order+1, dtype=self.ftype)
        for start in range(0, nc, batch_size):
            index = slice(start, min(start+batch_size, nc))
            point = self.mesh.bc_to_point(bcs, index=index)
            for s in range(order+1):
                if s == 0:
                    value = bm.einsum('qi,ci->cq', basis,
                                      coefficient[index])
                    square = (value-exact[s](point))**2
                else:
                    derivative, geometry, multiplicity = derivatives[s-1]
                    lambda_value = bm.einsum('aqi,ci->cqa', derivative,
                                             coefficient[index])
                    value = bm.einsum('cqa,acg->cqg', lambda_value,
                                      geometry[:, index])
                    difference = value-exact[s](point)
                    square = bm.einsum('cqg,g,cqg->cq', difference,
                                       multiplicity, difference)
                integral = bm.einsum('cq,q,c->', square, weights,
                                     measure[index])
                errors = bm.set_at(errors, s, errors[s]+integral)
        return bm.sqrt(errors)
