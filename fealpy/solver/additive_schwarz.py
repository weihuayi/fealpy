from fealpy.solver import cg,GAMGSolver
from fealpy.operator import LinearOperator
from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix
bm.set_backend('numpy')



class AdditiveSchwarz:
    
    def __init__(self, space, A, spacetype ='H1',type='vertex' ):

        self.space = space
        self.mesh = space.mesh
        self.A = A
        self.spacetype = spacetype
        self.type = type
        self.pre = self.generate_preconditioner()
        

    def solve(self, r):
        A = self.A
        e ,info = cg(A, r, M = self.pre,returninfo=True)
        print("CG solver info:", info)

        return e
    
    def build_face_patch_dofs(self):

        f2c = self.mesh.face_to_cell()    
        GD = self.mesh.GD      
        ledof = self.space.number_of_local_dofs(doftype='edge')

        c2d = self.space.dof.cell_to_dof()      # (NC, ldof_total)
        e2d = self.space.dof.edge_to_dof()      # (NF, ledof)  (2D里face=edge)

        cells = f2c[:, :2]
        c2d_cell = c2d[:, GD * ledof:]      # (NC, cdof)

        cell_dofs = c2d_cell[cells].reshape(cells.shape[0], -1)
        all_dofs = bm.concatenate([cell_dofs, e2d], axis=1)
        patch_dofs = [bm.unique(row) for row in all_dofs]
        
        return patch_dofs
    
    def build_vertex_patch_dofs(self):

        node2cell = self.mesh.node_to_cell()
        node2edge = self.mesh.node_to_edge()

        cell2dof = self.space.dof.cell_to_dof()
        edge2dof = self.space.dof.edge_to_dof()

        ledof = self.space.number_of_local_dofs(doftype='edge')
        cell2dof = cell2dof[:, self.mesh.GD * ledof:]  

        NN = node2cell.shape[0]
        patch_dofs = []
        for v in range(NN):
            cells = node2cell.indices[node2cell.indptr[v]:node2cell.indptr[v+1]]   # adj cells of node v
            edges = node2edge.indices[node2edge.indptr[v]:node2edge.indptr[v+1]]   # adj edges of node v

            dofs_from_cells = cell2dof[cells].ravel() 
            dofs_from_edges = edge2dof[edges].ravel() 

            patch = bm.unique(bm.concatenate((dofs_from_cells, dofs_from_edges)))
            patch_dofs.append(patch)

        return patch_dofs


    def preconditioner(self):
        """
        Vertex-based additive Schwarz preconditioner
        """

        if self.type == 'face':
            patch_dofs = self.build_face_patch_dofs()
            N = self.mesh.number_of_faces()
        elif self.type =='vertex':
            patch_dofs = self.build_vertex_patch_dofs()
            N = self.mesh.number_of_nodes()

        A_diag = []
        for v, dofs in enumerate(patch_dofs):
            
            A_sub = self.A[dofs, :][:, dofs]   
            A_sub_dense = A_sub.toarray()

            invA = bm.linalg.inv(A_sub_dense)  # 小块才建议直接求逆
            A_diag.append((invA, dofs))

        def precondition(r):
            r = r.astype(bm.float64)
            e = bm.zeros_like(r)
            for i in range(N):
                A_i, dofs = A_diag[i]
                e[dofs] += A_i @ r[dofs]
            return e
        return LinearOperator(shape=self.A.shape, matvec=precondition)

    def generate_preconditioner(self):
        A     = self.A
        space = self.space
        
        if self.spacetype == 'Hdiv':
            from fealpy.functionspace import RaviartThomasFESpace
            space0 = RaviartThomasFESpace(space.mesh, p=0)
        elif self.spacetype == 'H1':
            from fealpy.functionspace import LagrangeFESpace
            space0 = LagrangeFESpace(space.mesh, p=1)
        elif self.spacetype == 'Hcurl':
            from fealpy.functionspace import FirstNedelecFESpace
            space0 = FirstNedelecFESpace(space.mesh, p=0)
            
        PI = self.projection(space0)

        A0 = ((PI.T)@A@PI)
        mask = bm.abs(A0.data) < 1e-15
        A0.data[mask] = 0
        A0.sum_duplicates()

        P1 = GAMGSolver()
        P1.setup(A0)
        P0 = self.preconditioner()

        def mix_preconditioner(r1, P0, P1, PI):
            
            r1  = r1.astype(bm.float64)
            r,_ = P1.solve(PI.T@r1)
            print(r)
            e1  = PI@r
            e1 += 0.5*(P0@r1)
            return e1 

        pre_fun = lambda r: mix_preconditioner(r, P0, P1, PI)
        pre = LinearOperator(shape=A.shape, matvec=pre_fun)
        
        return pre

    def projection(self, space0):
        """
        L2 projection from space0 to space1
        @TODO: There exists a more efficient implementation.
        """

        space = self.space
        p = space.p
        q = p + 3

        mesh = space0.mesh

        qf = mesh.quadrature_formula(q, "cell") 
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi  = space.basis(bcs) # (NC, NQ, ldof0)
        phi0 = space0.basis(bcs)

        cm = mesh.entity_measure("cell")
        if self.spacetype == 'Hdiv' or self.spacetype == 'Hcurl':
            F  = bm.einsum('cqld, cqmd, q, c-> clm', phi0, phi, ws, cm)
            M  = bm.einsum('cqld, cqmd, q, c-> clm', phi, phi, ws, cm)
        elif self.spacetype == 'H1':
            F  = bm.einsum('cql, cqm, q, c-> clm', phi0, phi, ws, cm)
            M  = bm.einsum('cql, cqm, q, c-> clm', phi, phi, ws, cm)
            
        Minv = bm.linalg.inv(M)
        F = bm.einsum('clm, cmd->cld', F, Minv)

        c2d0   = space0.cell_to_dof()
        gdof0  = space0.number_of_global_dofs()
        c2d  = space.cell_to_dof()
        gdof = space.number_of_global_dofs()

        I = bm.broadcast_to(c2d[:, None], F.shape).reshape(-1)
        J = bm.broadcast_to(c2d0[..., None], F.shape).reshape(-1)
        data = F.reshape(-1)

        IJ = bm.concatenate([I[None, :], J[None, :]], axis=0)
        unique_IJ, index = bm.unique(IJ, axis=1, return_index=True)

        unique_data = data[index]
        M = csr_matrix((unique_data, (unique_IJ[0], unique_IJ[1])), 
                       shape=(gdof, gdof0), dtype=phi.dtype)
        
        return M

from fealpy.functionspace import TensorFunctionSpace, LagrangeFESpace,RaviartThomasFESpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarSourceIntegrator, ScalarNeumannBCIntegrator, ScalarMassIntegrator, GradPressureIntegrator,DivIntegrator,DivIntegrator2,DirichletBC    
from fealpy.solver import spsolve,GAMGSolver
from fealpy.model import PDEModelManager

pde = PDEModelManager('darcyforchheimer').get_example(9)
mesh = pde.init_mesh['uniform_tri'](nx=8, ny=8)
p = 6
q = p+3
unit = mesh.edge_unit_normal()
uspace = RaviartThomasFESpace(mesh, p=p)
u_bform = BilinearForm(uspace)
Mu = ScalarMassIntegrator(coef=1,q=q)
u_bform.add_integrator(Mu)
u_bform.add_integrator(DivIntegrator2(coef=1,q=q))

M = u_bform.assembly()

ulform = LinearForm(uspace)
ulform.add_integrator(ScalarSourceIntegrator(pde.f, q=q))
b = bm.random.rand(M.shape[0]) 
F = M@b

Solver = AdditiveSchwarz(uspace, M, spacetype='Hdiv',type='face' )
b = Solver.solve(F)