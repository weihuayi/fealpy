import numpy as np

from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace as Space
from utilfuncs import compute_filter

class TopSimp:
    def __init__(self, mesh=None, space=None, bc=None, material=None, filter=None):

        # Default mesh parameters
        if mesh is None:
            nelx, nely = 3, 2
            domain = [0, 3, 0, 2]
            hx = (domain[1] - domain[0]) / nelx
            hy = (domain[3] - domain[2]) / nely
            mesh = UniformMesh2d(extent=(0, nelx, 0, nely), h=(hx, hy), origin=(domain[0], domain[2]))

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

        self.mesh = mesh
        self.space = space
        self.bc = bc
        self.material = material
        self.ft = ft

    # SIMP 材料插值模型
    def material_model(self, rho):
        E = self.material['E0'] * (rho) ** self.material['penal']
        return E

    def fe_analysis(self, rho):
        from fealpy.fem import LinearElasticityOperatorIntegrator
        from fealpy.fem import BilinearForm
        from scipy.sparse import spdiags
        from scipy.sparse.linalg import spsolve
        GD = 2
        uh = self.space.function(dim=GD)
        lambda_ = (self.material['E0'] * self.material['nu']) / ((1+self.material['nu']) * (1-2*self.material['nu']))
        mu = (self.material['E0']) / (2 * (1+self.material['nu']))
        
        p = 1
        integrator = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=p+1)
        vspace = GD * (self.space, )
        bform = BilinearForm(vspace)
        bform.add_domain_integrator(integrator)
        KK = integrator.assembly_cell_matrix(space=vspace)
        print("KK:", KK.shape, "\n", KK[0].round(4))
        K = bform.assembly()
        #print("K:", K.shape, "\n", K.toarray().round(4))

        dflag = self.bc['fixeddofs']
        #print("dflag:", dflag)
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

        return uh
    
    def compute_compliance(self, rho):
        pass
