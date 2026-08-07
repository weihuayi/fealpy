from fealpy.backend import backend_manager as bm
from fealpy.fem.integrator import LinearInt, OpInt, CellInt


class HyperElasticResidualIntegrator(LinearInt, OpInt, CellInt):

    def __init__(self, material, space, q=None):
        self.material = material
        self.space = space
        self.q = q

    def assembly_cell_vector(self, space = None, uh = None):


        if space is None:
            raise ValueError("space must be provided.")
        if uh is None:
            raise ValueError("uh must be provided.")

        mesh = space.mesh
        # ----------------------------------
        # Cell DOF mapping
        # ----------------------------------
        cell2dof = space.cell_to_dof()

        # (NC, tldof)
        ue = uh[cell2dof]

        NC = ue.shape[0]
        GD = mesh.geo_dimension()

        # ----------------------------------
        # Quadrature
        # ----------------------------------
        q = space.p + 3 if self.q is None else self.q

        qf = mesh.quadrature_formula(q)

        bcs, ws = qf.get_quadrature_points_and_weights()

        # ----------------------------------
        # Shape function gradients
        # ----------------------------------
        gphi = space.scalar_space.grad_basis(
            bcs,
            variable='x'
        )

        # gphi:
        # (NC, NQ, ldof, GD)

        ldof = gphi.shape[2]


        ue = ue.reshape(NC, 
                        GD,
                        ldof
                        )
        # ue:
        # (NC, GD, ldof)
        # ----------------------------------

        grad_u = bm.einsum(
            'cia,cqaj->cqij',
            ue,
            gphi
        )

        # grad_u:
        # (NC,NQ,GD,GD)

        I = bm.eye(
            GD,
            dtype=grad_u.dtype
        )

        F = grad_u + I

        # ----------------------------------
        # First Piola stress
        # ----------------------------------

        P = self.material.stress(F)

        # P:
        # (NC,NQ,GD,GD)

        # ----------------------------------
        # Cell measure
        # ----------------------------------

        cm = mesh.entity_measure('cell')

        # ----------------------------------
        # Residual
        #
        # R_ai
        # =
        # ∫ P_ij dNa/dXj dV
        # ----------------------------------

        Re = bm.einsum(
            'q,c,cqij,cqaj->cai',
            ws,
            cm,
            P,
            gphi
        )

        # (NC, ldof, GD)

        # ----------------------------------
        # Convert to dof_priority=True
        # ----------------------------------

        Re = bm.swapaxes(Re, 1, 2)

        # (NC, GD, ldof)

        Re = Re.reshape(NC, GD*ldof)
        return Re