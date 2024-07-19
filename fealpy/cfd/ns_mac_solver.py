import numpy as np
from scipy.sparse import diags
from scipy.sparse import diags, lil_matrix,csr_matrix,coo_matrix
from scipy.sparse import vstack,hstack
from scipy.sparse.linalg import spsolve
from ..mesh import UniformMesh2d


class NSMacSolver():
    def __init__(self,Re, mesh):
        self.ftype = np.float64
        self.mesh = mesh
        nx = int(mesh.ds.nx)
        ny = int(mesh.ds.ny)
        hx = mesh.h[0]
        hy = mesh.h[1]
        self.umesh = UniformMesh2d([0, nx, 0, ny-1], h=(hx, hy), origin=(0, 0+hy/2))
        self.vmesh = UniformMesh2d([0, nx-1, 0, ny], h=(hx, hy), origin=(0+hx/2, 0))
        self.pmesh = UniformMesh2d([0, nx-1, 0, ny-1], h=(hx, hy), origin=(0+hx/2, 0+hy/2))

    def du(self):
        mesh = self.umesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(2*mesh.h[0])
        cy = 1/(2*mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A = diags([-cx,cx],[-n1,n1],shape=(NN,NN),format='csr')
        B = diags([0,cy,-cy],[0,1,-1],shape=(NN,NN),format='csr')

        val0 = np.broadcast_to(2*cy,(n0,))
        I0 = k[:,0]
        I_0 = k[:,-1]
        B += csr_matrix((val0,(I0,I0)),shape=(NN,NN),dtype=self.ftype)
        B += csr_matrix((-val0,(I_0,I_0)),shape=(NN,NN),dtype=self.ftype)
        
        val1 = np.broadcast_to(-cy/3,(n0,))
        J1 = k[:,1]
        J_1 = k[:,-2]
        B += csr_matrix((val1,(I0,J1)),shape=(NN,NN),dtype=self.ftype)
        B += csr_matrix((-val1,(I_0,J_1)),shape=(NN,NN),dtype=self.ftype)
        
        val2 = np.broadcast_to(cy,(n0-1,))
        J0m = k[1:,0]
        J1m = k[:-1,-1]
        JNm = k[:-1,-1]
        B += csr_matrix((val2,(J0m,JNm)),shape=(NN,NN),dtype=self.ftype)
        B += csr_matrix((-val2,(J1m,J0m)),shape=(NN,NN),dtype=self.ftype)
        return A,B

    def dv(self):
        mesh = self.vmesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(2*mesh.h[0])
        cy = 1/(2*mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A = diags([0,cx,-cx],[0,n1,-n1],shape=(NN,NN), format='csr')
        B = diags([0,cx,-cx],[0,1,-1],shape=(NN,NN), format='csr')
        
        val0 = np.broadcast_to(2*cx,(n1,))
        val1 = np.broadcast_to(-cx/3,(n1,))
        I0 = k[0,:]
        I_0 = k[-1,:]
        J1 = k[1,:]
        J_1 = k[-2,:]
        A += csr_matrix((val0, (I0,I0)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0,J1)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((-val0, (I_0,I_0)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((-val1, (I_0,J_1)), shape=(NN,NN), dtype=self.ftype)   
        return A,B
    
    def Tuv(self):
        vmesh = self.vmesh
        umesh = self.umesh
        vn0 = vmesh.ds.nx+1 
        vn1 = vmesh.ds.ny+1 
        un0 = umesh.ds.nx+1 
        un1 = umesh.ds.ny+1 
        NN = umesh.number_of_nodes()
        vk = np.arange(NN).reshape(vn0,vn1)
        uk = np.arange(NN).reshape(un0,un1)

        val = np.broadcast_to(1/4, (NN-2*un1,))
        Iu = uk[1:un0-1,:].flat
        Jv0 = vk[:vn0-1,1:].flat
        Jv1 = vk[1:,1:].flat
        Jv2 = vk[1:,:vn1-1].flat
        Jv3 = vk[:vn0-1,:vn1-1].flat
        A = csr_matrix((val, (Iu, Jv0)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iu, Jv1)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iu, Jv2)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iu, Jv3)), shape=(NN, NN), dtype=self.ftype)
        return A
    
    def Tvu(self):
        vmesh = self.vmesh
        umesh = self.umesh
        vn0 = vmesh.ds.nx+1 
        vn1 = vmesh.ds.ny+1 
        un0 = umesh.ds.nx+1 
        un1 = umesh.ds.ny+1 
        NN = umesh.number_of_nodes()
        vk = np.arange(NN).reshape(vn0,vn1)
        uk = np.arange(NN).reshape(un0,un1)
        
        val = np.broadcast_to(1/4, (NN-2*vn0,))
        Iv = vk[:,1:vn1-1].flat
        Ju0 = uk[:un0-1,1:].flat
        Ju1 = uk[1:,1:].flat
        Ju2 = uk[1:,:un1-1].flat
        Ju3 = uk[:un0-1,:un1-1].flat
        A = csr_matrix((val, (Iv, Ju0)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iv, Ju1)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iv, Ju2)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (Iv, Ju3)), shape=(NN, NN), dtype=self.ftype)
        return A
        
    def laplace_u(self,c=None):
        mesh = self.umesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(mesh.h[0])
        cy = 1/(mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A = diags([-4*cx*cy,cx*cy,cx*cy,cx*cy,cx*cy], [0,1,-1,n1,-n1], shape=(NN, NN), format='csr')
        
        val0 = np.broadcast_to(-2*cx*cy,(n0,))
        I0 = k[:,0]
        I_0 = k[:,-1]
        A += csr_matrix((val0, (I0,I0)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val0, (I_0,I_0)), shape=(NN,NN), dtype=self.ftype)
        
        val1 = np.broadcast_to((1/3)*cx*cy,(n0,))
        J1 = k[:,1]
        J_1 = k[:,-2]
        A += csr_matrix((val1, (I0,J1)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val1, (I_0,J_1)), shape=(NN,NN), dtype=self.ftype)
        
        val2 = np.broadcast_to(-cx*cy,(n0-1,))
        I0m = k[1:,0]
        J_0m = k[:-1,-1]
        J0m = k[1:,0]
        A += csr_matrix((val2, (I0m,J_0m)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val2, (J_0m,J0m)), shape=(NN,NN), dtype=self.ftype)
        if c == None:
            return A
        else:
            return c*A

    def laplace_v(self,c=None):
        mesh = self.vmesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(mesh.h[0])
        cy = 1/(mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A = diags([-4*cx*cy,cx*cy,cx*cy,cx*cy,cx*cy], [0,1,-1,n1,-n1], shape=(NN, NN), format='csr')

        val0 = np.broadcast_to(-2*cx*cy, (n1,))
        val1 = np.broadcast_to((1/3)*cx*cy, (n1,))
        I0 = k[0,:]
        I_0 = k[-1,:]
        J0m = k[1,:]
        J_0m = k[-2,:]
        A += csr_matrix((val0, (I0,I0)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val0, (I_0,I_0)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0,J0m)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val1, (I_0,J_0m)), shape=(NN,NN), dtype=self.ftype)
        if c == None:
            return A
        else:
            return c*A
    
    def dp_u(self):
        mesh = self.pmesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(mesh.h[0])
        cy = 1/(mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A0 = diags([cx,-cx], [0,-n1], shape=(NN, NN), format='csr')
        A1 = diags([0], [0], shape=(n1, NN), format='csr')
        A = vstack([A0, A1], format='csr')  
        A[ :n1, : ] = 0 
        return A
    
    def dp_v(self):
        vmesh = self.vmesh
        pmesh = self.pmesh
        vn0 = vmesh.ds.nx+1 
        vn1 = vmesh.ds.ny+1 
        pn0 = pmesh.ds.nx+1 
        pn1 = pmesh.ds.ny+1 
        NNv = vmesh.number_of_nodes()
        NNp = pmesh.number_of_nodes()
        cx = 1/(pmesh.h[0])
        cy = 1/(pmesh.h[1])
        vk = np.arange(NNv).reshape(vn0,vn1)
        pk = np.arange(NNp).reshape(pn0,pn1)

        val = np.broadcast_to(cy, (NNv-2*vn0,))
        Iv = vk[:,1:vn1-1].flat
        Jp0 = pk[:,:pn1-1].flat
        Jp1 = pk[:,1:].flat
        A = csr_matrix((val, (Iv, Jp1)), shape=(NNv, NNp), dtype=self.ftype)
        A += csr_matrix((-val, (Iv, Jp0)), shape=(NNv, NNp), dtype=self.ftype)
        return A
    
    def source_Fx(self, pde ,t):
        mesh = self.umesh
        nodes = mesh.entity('node')
        source = pde.source_F(nodes,t) 
        return source
    
    def dpm(self):
        vmesh = self.vmesh
        umesh = self.umesh
        pmesh = self.pmesh
        cx = 1 / (umesh.h[0])
        cy = 1 / (vmesh.h[1])
        vn0 = vmesh.ds.nx+1 
        vn1 = vmesh.ds.ny+1 
        un0 = umesh.ds.nx+1 
        un1 = umesh.ds.ny+1 
        pn0 = pmesh.ds.nx+1 
        pn1 = pmesh.ds.ny+1 
        NN = umesh.number_of_nodes()
        NNp = pmesh.number_of_nodes()
        vk = np.arange(NN).reshape(vn0,vn1)
        uk = np.arange(NN).reshape(un0,un1)
        pk = np.arange(NNp).reshape(pn0,pn1)
        A = diags([-cx,cx], [0,un1], shape=(NNp, NN), format='csr')

        val = np.broadcast_to(cy, (NN-vn0,))
        I = pk.flat
        J0m = vk[:,:-1].flat
        J_0m = vk[:,1:].flat
        B = csr_matrix((val, (I,J_0m)), shape=(NNp, NN), dtype=self.ftype)
        B += csr_matrix((-val, (I,J0m)), shape=(NNp, NN), dtype=self.ftype)
        return A,B

    def laplace_phi(self):
        mesh = self.pmesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        cx = 1/(mesh.h[0])
        cy = 1/(mesh.h[1])
        NN = mesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        A = diags([-4*cx*cy,cx*cy,cx*cy,cx*cy,cx*cy], [0,1,-1,n1,-n1], shape=(NN, NN), format='csr')
        
        val0 = np.broadcast_to(2*cx*cy,(4,))
        val1 = np.broadcast_to(1*cx*cy,(n1-2,))
        val2 = np.broadcast_to(-1*cx*cy,(n1-1,))
        I0c = [k[0,0],k[0,-1],k[-1,0],k[-1,-1]]
        I0l = k[0,1:n1-1]
        I0r = k[-1,1:n1-1]
        I0d = k[1:n1-1,0]
        I0u = k[1:n1-1,-1]
        J0m = k[1:,0]
        J_0m = k[:-1,-1]
        A += csr_matrix((val0, (I0c, I0c)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0l, I0l)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0r, I0r)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0d, I0d)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val1, (I0u, I0u)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val2, (J_0m, J0m)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val2, (J0m, J_0m)), shape=(NN, NN), dtype=self.ftype)
        return A.toarray()

    #找v网格边界点位置
    def vnodes_ub(self):
        mesh = self.mesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        vmesh = self.vmesh
        vn0 = vmesh.ds.nx+1 
        vn1 = vmesh.ds.ny+1 
        nodes = mesh.entity('node')
        vnodes = vmesh.entity('node')
        NN = mesh.number_of_nodes()
        NNv = vmesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        kv = np.arange(NNv).reshape(vn0,vn1)
        A0 = nodes[1:vn1-1,:] 
        A1 = nodes[NN-vn1+1:-1,:]
        I0 = kv[0,1:vn1-1]
        I1 = kv[-1,1:vn1-1]
        vnodes[I0,[0]] = A0[:,0]
        vnodes[I1,[0]] = A1[:,0]
        vnodes[kv[1:-1,:],0] = 0
        vnodes[kv[1:-1,:],1] = 0
        return vnodes

    #找u网格边界点位置
    def unodes_ub(self):
        mesh = self.mesh
        n0 = mesh.ds.nx+1 
        n1 = mesh.ds.ny+1 
        umesh = self.umesh
        un0 = umesh.ds.nx+1 
        un1 = umesh.ds.ny+1 
        nodes = mesh.entity('node')
        unodes = umesh.entity('node')
        NN = mesh.number_of_nodes()
        NNu = umesh.number_of_nodes()
        k = np.arange(NN).reshape(n0,n1)
        ku = np.arange(NNu).reshape(un0,un1)
        I_0 = k[1:n0-1,0]
        I_1 = k[1:n0-1,-1]
        A0 = nodes[I_0,:]
        A1 = nodes[I_1,:]
        I0 = ku[1:un0-1,0]
        I1 = ku[1:un0-1,-1]
        unodes[I0,1] = A0[:,1]
        unodes[I1,1] = A1[:,1]
        unodes[ku[:,1:-1],0] = 0
        unodes[ku[:,1:-1],1] = 0
        return unodes