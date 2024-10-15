from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from fealpy.mesh import TriangleMesh

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (BilinearForm,
                                     LinearForm,
                                     ScalarDiffusionIntegrator,
                                     ScalarSourceIntegrator,
                                     DirichletBC)

from .. import logger


class MovingMeshAlg:
    def __init__(self, mesh ,p, pde , uh = None ,  model = '1'):
        self.mesh = mesh
        self.p = p 
        self.space = LagrangeFESpace(self.mesh, p)
        self.cell = mesh.cell
        self.uh = uh
        self.pde = pde
        self.model = model
        self.isbdnode = mesh.boundary_node_flag()


    def __call__(self, node: TensorLike):

        if self.model == '1':
            E = self.get_energy(node)
            grad = self.grad_E(node)
            
        if self.model == '2':
            node.requires_grad_(True) 
            E = self.get_energy(node)
            node.retain_grad()
            E.backward(retain_graph=True)
            grad = node.grad

        return E , grad
    
    def get_energy(self,node : TensorLike):
        p =1
        cell = self.mesh.cell
        isbdnode = self.isbdnode
        source = self.pde.source

        mesh = TriangleMesh(node,cell)     
        space = LagrangeFESpace(mesh,p)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(method="fast"))
        A = bform.assembly()
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(source))
        F = lform.assembly()
        # 去除边界项的影响
        A = A.to_dense()
        A = A[~isbdnode]
        A = A[:,~isbdnode]
        F = F[~isbdnode]

        uh = self.pde.solution(node)[~isbdnode]  

        E = 1/2 * bm.einsum('i , ij ,j->', uh, A, uh) - bm.dot(uh,F)
        return E
    
    def get_energy_local(self):
        mesh = self.mesh
        node = mesh.entity('node')
        uh = self.pde.solution(node)

        p = self.p
   
        tau = mesh.entity_measure('cell')
        # print(tau)
        qf = mesh.integrator(p+1)
        bcs,ws = qf.get_quadrature_points_and_weights()
        gphi = self.space.grad_basis(bcs)
        ps = mesh.bc_to_point(bcs)
        fval = self.pde.source(ps)
        phi = self.space.basis(bcs)
        cell2dof = self.space.cell_to_dof()

        A_k = bm.einsum('l , klid , kljd -> kij',ws,gphi,gphi)
        F_k = bm.einsum('l , kl , kli , k -> ki',ws,fval,phi,tau)
        I_k = 1/2 * bm.einsum('ki , kij , kj -> k',uh[cell2dof], A_k , uh[cell2dof])
        J_k = bm.einsum('ki , ki -> k',uh[cell2dof],F_k)
        return I_k , J_k
    
    def grad_tau(self):
        mesh = self.mesh
        node = mesh.entity("node")
        cell = self.cell
        roll_front = node[cell][:,[1,2,0]]
        roll_back = node[cell][:,[2,0,1]]

        W = bm.array([[0,1],[-1,0]])
        G1 = bm.einsum('knd , de -> kne',roll_back,W)#(NC,LDOF,GD) 
        G2 = bm.einsum('knd , de -> kne',roll_front,W.T)
        g_tau = 1/2*(G1 + G2)

        return g_tau
    
    def grad_IJ(self,node): 
        pde = self.pde
        cell = self.cell
        p = self.p
        self.mesh = TriangleMesh(node , cell)
        self.space = LagrangeFESpace(self.mesh, p)
        mesh = self.mesh

        u = self.pde.solution(node)

        I_k,J_k = self.get_energy_local()
        print("J_k:",J_k)
 
        g_tau = self.grad_tau()
        tau = mesh.entity_measure('cell') 
        print("1/tau:\n",1/tau)

        cell2dof = self.space.cell_to_dof()

        delta_n = bm.array([[0,1,-1],[-1,0,1],[1,-1,0]])
    
        roll_front = node[cell][:,[1,2,0]]
        roll_back = node[cell][:,[2,0,1]]

        cell2nodes_m = roll_back - roll_front

        DX = bm.einsum('ni , kjd -> nikjd', delta_n , cell2nodes_m)
        DXT = bm.transpose(DX , (0,3,2,1,4))
        DXDX = DX + DXT
        I_xk = 1/8 * bm.einsum(' ki ,nikjd , kj -> knd', u[cell2dof], DXDX , u[cell2dof])

        I_x = bm.zeros(node.shape,dtype=bm.float64)
        bm.add_at(I_x,cell,I_xk)


        p = self.p
        qf = mesh.integrator(p+1)
        bcs,ws = qf.get_quadrature_points_and_weights()
        phi = self.space.basis(bcs)
        ps = mesh.bc_to_point(bcs)
        fval = self.pde.source(ps)
        gphi = self.space.grad_basis(bcs)

        J_x1 = 2*J_k[:,None,None] * g_tau #(NC,LODF,GD)
        # print("J_x1:",J_x1)
        # ps 表示每个积分点在每个单元上的坐标
        g_f = pde.grad_f(ps)#(NQ,NC,GD)

        gfl = bm.einsum('kld , kli -> klid',g_f , phi)
        fgl = bm.einsum('kl, klid -> klid',fval , gphi)

        J_x2 = bm.einsum('k , ki , l , ln , klid -> knd ',tau**2 ,u[cell2dof] , ws , bcs , gfl+fgl )
        J_xk = J_x1 + J_x2
        J_x = bm.zeros(node.shape,dtype=bm.float64)
        bm.add_at(J_x,cell,J_xk)

        return I_x ,J_x
    
    def grad_E(self,node):

        isbdnode = self.isbdnode
        grad_I , grad_J = self.grad_IJ(node)
        grad_E = grad_I - grad_J
        grad_E = (1 - isbdnode)[:,None] * grad_E
        return grad_E
    


