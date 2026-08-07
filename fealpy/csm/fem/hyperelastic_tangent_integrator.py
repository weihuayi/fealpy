from fealpy.backend import backend_manager as bm
from fealpy.fem.integrator import LinearInt, OpInt, CellInt


class HyperElasticTangentIntegrator(LinearInt, OpInt, CellInt):

    def __init__(self, material, space, q=None):
        self.material = material
        self.space = space
        self.q = q

    def assembly_cell_matrix(self, space, uh):

        mesh = space.mesh
        scalar_space = space.scalar_space

        # -----------------------------
        # DOF mapping
        # -----------------------------
        cell2dof = space.cell_to_dof()
        ue = uh[cell2dof]

        NC = ue.shape[0]
        GD = mesh.geo_dimension()

        # -----------------------------
        # Quadrature
        # -----------------------------
        q = scalar_space.p + 3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # -----------------------------
        # Shape gradients
        # -----------------------------
        gphi = scalar_space.grad_basis(bcs, variable='x')

        # -----------------------------
        # reshape displacement (not needed for K, but consistent style)
        # -----------------------------
        ue = ue.reshape(NC, GD, -1)

        # -----------------------------
        # Deformation gradient
        # -----------------------------
        grad_u = bm.einsum('cia,cqaj->cqij', ue, gphi)
        F = grad_u + bm.eye(GD, dtype=grad_u.dtype)

        # -----------------------------
        # Material tangent
        # -----------------------------
        A = self.material.tangent(F)

        # -----------------------------
        # Cell measure
        # -----------------------------
        cm = mesh.entity_measure('cell')

        # -----------------------------
        # Assembly stiffness matrix
        # -----------------------------
        Ke = bm.einsum(
            'q,c,cqijkl,cqaj,cqbl->caibk',
            ws,
            cm,
            A,
            gphi,
            gphi
        )
        
        Ke = bm.swapaxes(Ke, 1, 2)
        Ke= bm.swapaxes(Ke, 3, 4)


        NC = Ke.shape[0]
        GD = Ke.shape[1]
        ldof = Ke.shape[2]

        tldof = GD * ldof

        Ke = Ke.reshape(NC, GD*ldof, GD*ldof)

        return Ke

        