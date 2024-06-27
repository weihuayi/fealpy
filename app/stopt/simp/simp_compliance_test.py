import numpy as np

from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace as Space
from utilfuncs import compute_filter

class TopSimp:
    def __init__(self, mesh=None, space=None, bc=None, material=None, \
                filter=None, global_volume_constraints=None):

        # Default mesh parameters
        if mesh is None:
            nx, ny = 3, 2
            domain = [0, 3, 0, 2]
            hx = (domain[1] - domain[0]) / nx
            hy = (domain[3] - domain[2]) / ny
            mesh = UniformMesh2d(extent=(0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.gca()
            mesh.add_plot(axes)
            mesh.find_node(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

        # Default space parameters
        if space is None:
            p = 1
            GD = 2
            space = Space(mesh, p=p, doforder='vdims')
            vspace = GD * (space, )

        # Default boundary conditions and loads
        if bc is None:
            example = 1
            if example == 1:
                gdof = vspace[0].number_of_global_dofs()
                vgdof = gdof * GD
                force = np.zeros( (vgdof, 1) )
                force[vgdof-1, 0] = -1
                #force[1, 0] = -1
                fixeddofs = np.arange(0, 2*(mesh.ny+1), 1)
                # np.union1d( np.arange(0, 2*(nely+1), 2), np.array([2*(nelx+1)*(nely+1) - 1]) )

            bc = {'force': force, 'fixeddofs': fixeddofs}

        # Default material parameters
        if material is None:
            material = {'E0': 1., 'nu': 0.3, 'penal': 3.}

        # Default filter parameters
        if filter is None:
            filter_radius = 1.5
            H, Hs = compute_filter(mesh, filter_radius)
            ft = {'type':1, 'H':H, 'Hs':Hs}

        # Default global volume constraints
        if global_volume_constraints is None:
            global_volume_constraints = {'isOn':True, 'volfrac':0.5}

        self.mesh = mesh
        self.space = space
        self.bc = bc
        self.material = material
        self.ft = ft
        self.global_volume_constraints = global_volume_constraints

    # SIMP 材料插值模型
    def material_model(self, rho):
        E = self.material['E0'] * (rho) ** self.material['penal']
        return E

    def fe_analysis(self, rho):
        from linear_elasticity_operator_intergrator_test import BeamOperatorIntegrator
        from fealpy.fem import BilinearForm
        from scipy.sparse import spdiags
        from scipy.sparse.linalg import spsolve
        GD = 2
        uh = self.space.function(dim=GD)
        
        integrator = BeamOperatorIntegrator(rho=rho, penal=self.material['penal'], \
                                            nu=self.material['nu'], E0=self.material['E0'])
        vspace = GD * (self.space, )
        bform = BilinearForm(vspace)
        bform.add_domain_integrator(integrator)
        KK = integrator.assembly_cell_matrix(space=vspace)
        #print("KK:", KK.shape, "\n", KK[0].round(4))
        K = bform.assembly()
        #print("K:", K.shape, "\n", K.toarray().round(4))

        dflag = self.bc['fixeddofs']
        uh.flat[dflag] = 0

        F = self.bc['force']
        #print("F:", F.shape, "\n", F.T.round(4))
        F -= K@uh.reshape(-1, 1)
        F[dflag.flat] = uh.reshape(-1, 1)[dflag.flat]
        #print("F:", F.shape, "\n", F.T.round(4))

        bdIdx = np.zeros(K.shape[0], dtype=np.int_)
        bdIdx[dflag.flat] = 1

        D0 = spdiags(1-bdIdx, 0, K.shape[0], K.shape[0])
        D1 = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
        K = D0@K@D0 + D1
        #print("K:", K.shape, "\n", K.toarray().round(4))

        # 线性方程组求解
        uh.flat[:] = spsolve(K, F)

        ldof = self.space.number_of_local_dofs()
        vldof = GD * ldof
        cell2dof = vspace[0].cell_to_dof()
        print("cell2dof:", cell2dof)
        NC = self.mesh.number_of_cells()
        ue = np.zeros((NC, vldof))

        reshaped_uh = uh.reshape(-1)
        # 每个单元的自由度（每个节点两个自由度）
        updated_cell2dof = np.repeat(cell2dof*GD, GD, axis=1) + np.tile(np.array([0, 1]), (NC, ldof))
        print("updated_cell2dof:", updated_cell2dof.shape, "\n", updated_cell2dof)
        idx = np.array([0, 1, 4, 5, 6, 7, 2, 3], dtype=np.int_)
        # 用 Top 中的自由度替换 FEALPy 中的自由度
        updated_cell2dof = updated_cell2dof[:, idx]
        ue = reshaped_uh[updated_cell2dof]

        return uh, ue
    
    def compute_compliance(self, rho):

        uh, ue = self.fe_analysis(rho)
        c1 = np.dot(self.bc['force'].T, uh.flat)

        nu = self.material['nu']
        E0 = self.material['E0']
        k = np.array([1/2 - nu/6,   1/8 + nu/8,   -1/4 - nu/12, -1/8 + 3 * nu/8,
                -1/4 + nu/12,  -1/8 - nu/8,    nu/6,         1/8 - 3 * nu/8])
        KE = E0 / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
            ])

        temp1 = rho ** self.material['penal']
        temp2 = np.einsum('ij, jk, ki -> i', ue, KE, ue.T)
        c2 = np.einsum('i, j -> ', temp1, temp2)

        return c1, c2
    
    def compute_compliance_sensitivity(self, rho):

        uh, ue = self.fe_analysis(rho)

        nu = self.material['nu']
        E0 = self.material['E0']
        k = np.array([1/2 - nu/6,   1/8 + nu/8,   -1/4 - nu/12, -1/8 + 3 * nu/8,
                -1/4 + nu/12,  -1/8 - nu/8,    nu/6,         1/8 - 3 * nu/8])
        KE = E0 / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
            ])

        temp1 = -self.material['penal'] * rho ** (self.material['penal'] - 1)
        temp2 = np.einsum('ij, jk, ki -> i', ue, KE, ue.T)
        dc = np.einsum('i, j -> ', temp1, temp2)

        return dc
    
    def compute_global_volume_constraint(self, rho):
        g = np.mean(rho) / self.global_volume_constraints['vf'] - 1.

        return g
