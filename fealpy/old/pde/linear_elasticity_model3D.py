import sympy as sp
import numpy as np

from fealpy.decorator  import cartesian 
from fealpy.pde.linear_elasticity_model import QiModel3d, PolyModel3d, Model2d, HuangModel2d
from sympy.calculus.util import not_empty_in

class LinearElasticityTempalte():
    def __init__(self):
        pass

    def domain(self):
        pass

    def init_mesh(self):
        pass

    @cartesian
    def displacement(self, p):
        pass

    @cartesian
    def jacobian(self, p):
        pass

    @cartesian
    def strain(self, p):
        pass

    @cartesian
    def stress(self, p):
        pass

    @cartesian
    def source(self, p):
        pass

    @cartesian
    def dirichlet(self, p):
        pass

    @cartesian
    def neumann(self, p):
        pass

    @cartesian
    def is_dirichlet_boundary(self, p):
        pass

    @cartesian
    def is_neuman_boundary(self, p):
        pass

    @cartesian
    def is_fracture_boundary(self, p):
        pass





class GenLinearElasticitymodel3D():
    def __init__(self,u,x,lam=1,mu=0.5,
                    Dirichletbd_n = None, Dirichletbd_t0 = None, Dirichletbd_t1 = None,
                    Neumannbd_nn = None, Neumannbd_nt0=None, Neumannbd_nt1=None):
        '''
        生成一般线弹性模型,应力边界条件法向和切向(2个方向)分开 3D
        '''
        dim = len(x)
        div_u = 0
        for i in range(dim):
            div_u+=sp.diff(u[i],x[i],1)

        stress = [[0 for i in range(dim)] for j in range(dim)]
        grad_displacement = [[0 for i in range(dim)] for j in range(dim)]

        for i in range(dim):
            for j in range(i,dim):
                if i == j:
                    stress[i][j] = mu*(sp.diff(u[i],x[j],1)+sp.diff(u[j],x[i],1))+lam*div_u
                else:
                    stress[i][j] = mu*(sp.diff(u[i],x[j],1)+sp.diff(u[j],x[i],1))
                    stress[j][i] = stress[i][j]

        for i in range(dim):
            for j in range(dim):
                grad_displacement[i][j] = sp.diff(u[i],x[j],1)

        


                

        source = [0 for i in range(dim)]

        for i in range(dim):
            for j in range(dim):
                source[i] -= sp.diff(stress[i][j],x[j],1)

        
        self.dim = dim
        self.lam = lam
        self.mu = mu

        self.displacements = np.array(u) 
        self.grad_displacements = np.array(grad_displacement)
        self.sources = np.array(source)
        self.stresss = np.array(stress)

        self.is_dirichletbd_n = Dirichletbd_n
        self.is_dirichletbd_t0 = Dirichletbd_t0
        self.is_dirichletbd_t1 = Dirichletbd_t1

        self.is_neumannbd_nn = Neumannbd_nn
        self.is_neumannbd_nt0 = Neumannbd_nt0
        self.is_neumannbd_nt1 = Neumannbd_nt1
        self.x = x

    def domain(self):
        return np.array([0,1,0,1,0,1])


    def init_mesh(self, n=0):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        domain = self.domain()
        mesh = mf.boxmesh3d(domain,nx=1,ny=1,nz=1,meshtype='tet')
        mesh.uniform_refine(n=n)
        return mesh

    @cartesian
    def displacement(self, p):
        return(self.Numpyfunction(self.displacements,p))
                
    @cartesian
    def grad_displacement(self,p):
        return(self.Numpyfunction(self.grad_displacements,p))

    @cartesian
    def stress(self, p):
        return(self.Numpyfunction(self.stresss,p))

    @cartesian
    def div_stress(self,p):
        return -self.source(p)


    @cartesian
    def source(self, p):
        return self.Numpyfunction(self.sources,p)

    @cartesian
    def dirichlet(self, p, n=None, t0=None, t1=None):
        displacement = self.displacement(p) #(NQ,NEbd,gdim)
        shape = p.shape
        val = np.zeros(shape,dtype=np.float)
        if len(shape) >=3:
            bd_type_idx = self.is_dirichlet_boundary(np.mean(p,axis=0))
        else:
            bd_type_idx = self.is_dirichlet_boundary(p)
        idx_n, = np.nonzero(bd_type_idx[0])
        if len(idx_n)>0:
            val[...,idx_n,:] += np.einsum('...i,...i,...j->...j',displacement[...,idx_n,:],n[idx_n],n[idx_n]) 
        
        idx_t0, = np.nonzero(bd_type_idx[1])
        if len(idx_t0)>0:
            val[...,idx_t0,:] += np.einsum('...i,...i,...j->...j',displacement[...,idx_t0,:],t0[idx_t0],t0[idx_t0])


        idx_t1, = np.nonzero(bd_type_idx[2])
        if len(idx_t1)>0:
            val[...,idx_t1,:] += np.einsum('...i,...i,...j->...j',displacement[...,idx_t1,:],t1[idx_t1],t1[idx_t1])
        return val
        

    @cartesian
    def neumann(self, p, n=None, t0=None,t1=None):
        stress = self.stress(p) #(NEbd,ldof,gdim,gdim)
        stress_n = np.einsum('...ij,...j->...i',stress,n) #(NEbd,ldof,2)
        shape = p.shape
        val = np.zeros(shape,dtype=np.float)

        if len(shape) >= 3:
            bd_type_idx = self.is_neumann_boundary(np.mean(p,axis=1))
        else:
            bd_type_idx = self.is_neumann_boundary(p)
        idx_n, = np.nonzero(bd_type_idx[0])

        if len(idx_n)>0:
            val[idx_n] += np.einsum('...i,...i,...j->...j',stress_n[idx_n],n[idx_n],n[idx_n]) 
        
        idx_t, = np.nonzero(bd_type_idx[1])
        if len(idx_t)>0:
            val[idx_t] += np.einsum('...i,...i,...j->...j',stress_n[idx_t],t0[idx_t],t0[idx_t])
        
        idx_t, = np.nonzero(bd_type_idx[2])
        if len(idx_t)>0:
            val[idx_t] += np.einsum('...i,...i,...j->...j',stress_n[idx_t],t1[idx_t],t1[idx_t])

        return val
        
    @cartesian
    def is_neumann_boundary(self, p):
        shape = (3,)+p.shape[:-1]
        is_neumannbd_idx = np.zeros(shape,dtype = np.bool_)
        x0 = p[...,0]
        x1 = p[...,1]
        x2 = p[...,2]
        if self.is_neumannbd_nn is not None:
            is_neumannbd_idx[0] = eval(self.is_neumannbd_nn)
        if self.is_neumannbd_nt0 is not None:
            is_neumannbd_idx[1] = eval(self.is_neumannbd_nt0)
        if self.is_neumannbd_nt1 is not None:
            is_neumannbd_idx[2] = eval(self.is_neumannbd_nt1)

        return is_neumannbd_idx 

    @cartesian
    def is_dirichlet_boundary(self, p):
        shape = (3,)+p.shape[:-1]
        is_dirichlet_idx = np.zeros(shape,dtype = np.bool_)
        x0 = p[...,0]
        x1 = p[...,1]
        x2 = p[...,2]
        if self.is_dirichletbd_n is not None:
            is_dirichlet_idx[0] = eval(self.is_dirichletbd_n)
        if self.is_dirichletbd_t0 is not None:
            is_dirichlet_idx[1] = eval(self.is_dirichletbd_t0)
        if self.is_dirichletbd_t1 is not None:
            is_dirichlet_idx[2] = eval(self.is_dirichletbd_t1)

        return is_dirichlet_idx

    
    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        dim = self.dim
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:dim], axis=-1)
        aphi[..., 0:dim] -= lam/(2*mu+dim*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi    


    def Numpyfunction(self,f,p):
        dim = self.dim
        shape = p.shape #(...,dim)
        if dim != shape[-1]:
            print('dimension is not right!')
            return
        else:
            sym_val = f
            val_dim = sym_val.shape
            x = self.x
            if len(val_dim) == 1:  
                shape = shape[:-1]
                shape+=val_dim
                val = np.zeros(shape,dtype=float)                             
                for i in range(val_dim[0]):
                        idx = False
                        for k in range(dim):
                            idx = (idx or (str(x[k]) in str(sym_val[i])))
                        if  idx:
                            lam_val = sp.lambdify(self.x,sym_val[i],'numpy')
                            if dim == 2:
                                val[...,i] = lam_val(p[...,0],p[...,1])
                            elif dim == 3:
                                val[...,i] = lam_val(p[...,0],p[...,1],p[...,2])
                        else:
                            #print(sym_val[i][j])
                            val[...,i]+=float(sym_val[i])
                return val
            elif len(val_dim) == 2:
                shape = shape[:-1]
                shape +=val_dim
                val = np.zeros(shape,dtype=float)   
                for i in range(val_dim[0]):
                    for j in range(val_dim[1]):
                        idx = False
                        for k in range(dim):
                            idx = (idx or (str(x[k]) in str(sym_val[i][j])))
                        if  idx:
                            lam_val = sp.lambdify(self.x,sym_val[i][j],'numpy')
                            if dim == 2:
                                val[...,i,j] = lam_val(p[...,0],p[...,1])
                            elif dim == 3:
                                val[...,i,j] = lam_val(p[...,0],p[...,1],p[...,2])
                        else:
                            #print(sym_val[i][j])
                            val[...,i,j]+=float(sym_val[i][j])
                return val
        




class GenLinearElasticitymodels():
    def __init__(self,u,x,lam=1,mu=0.5,stress_direction=None, Dirichletbd = None, Neumannbd = None, domain=None):
        '''
        生成一般线弹性模型
        '''
        dim = len(x)
        div_u = 0
        for i in range(dim):
            div_u+=sp.diff(u[i],x[i],1)

        stress = [[0 for i in range(dim)] for j in range(dim)]
        grad_displacement = [[0 for i in range(dim)] for j in range(dim)]

        for i in range(dim):
            for j in range(i,dim):
                if i == j:
                    stress[i][j] = mu*(sp.diff(u[i],x[j],1)+sp.diff(u[j],x[i],1))+lam*div_u
                else:
                    stress[i][j] = mu*(sp.diff(u[i],x[j],1)+sp.diff(u[j],x[i],1))
                    stress[j][i] = stress[i][j]

        for i in range(dim):
            for j in range(dim):
                grad_displacement[i][j] = sp.diff(u[i],x[j],1)

        


                

        source = [0 for i in range(dim)]

        for i in range(dim):
            for j in range(dim):
                source[i] -= sp.diff(stress[i][j],x[j],1)

        





        if domain is None:
            if dim == 2:
                domain = np.array([0,1,0,1])
            elif dim == 3:
                domain = np.array([0,1,0,1,0,1])

        self.dim = dim
        self.lam = lam
        self.mu = mu

        self.displacements = np.array(u) 
        self.grad_displacements = np.array(grad_displacement)
        self.sources = np.array(source)
        self.stresss = np.array(stress)

        self.domain = domain
        self.is_dirichletbd = Dirichletbd
        self.is_neumannbd = Neumannbd
        self.x = x

        self.stress_directions = stress_direction
        #self.stress_direction = 'stress_nt'
        #self.stress_direction = 'stress_nn'

    def domains(self):
        return self.domain


    def init_mesh(self, n=0):
        dim = self.dim
        if dim == 2:
            from fealpy.mesh import TriangleMesh
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1)], dtype=np.float)
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)

            mesh = TriangleMesh(node, cell)
        elif dim == 3:
            from fealpy.mesh import TetrahedronMesh
            node = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1], 
            [1, 1, 1],
            [0, 1, 1]], dtype=np.float) 

            cell = np.array([
                [0,1,2,6],
                [0,5,1,6],
                [0,4,5,6],
                [0,7,4,6],
                [0,3,7,6],
                [0,2,3,6]], dtype=np.int)
            mesh = TetrahedronMesh(node, cell)

        mesh.uniform_refine(n)
        return mesh 


    @cartesian
    def displacement(self, p):
        return(self.Numpyfunction(self.displacements,p))
                
    @cartesian
    def grad_displacement(self,p):
        return(self.Numpyfunction(self.grad_displacements,p))

    @cartesian
    def stress(self, p):
        return(self.Numpyfunction(self.stresss,p))

    @cartesian
    def div_stress(self,p):
        return -self.source(p)


    @cartesian
    def source(self, p):
        return self.Numpyfunction(self.sources,p)

    @cartesian
    def dirichlet(self, p, n=None, t=None):
        return self.displacement(p)

    @cartesian
    def neumann(self, p, n, t=None):
        stress = self.stress(p) #(...,gdim)
        stress_direction = self.stress_directions
        if stress_direction is None:
            val = np.einsum('...ij,...j->...i',stress,n)
        else:
            val = np.einsum('...i,...ij,...j->...',t,stress,n)
        return val
        
    @cartesian
    def is_neumann_boundary(self, p):
        if self.is_neumannbd is None:
            shape = p.shape[:-1]
            return np.zeros(shape,dtype = np.bool_)
        else:
            dim = self.dim
            if dim == 2:
                x0 = p[..., 0]
                x1 = p[..., 1]
            elif dim == 3:
                x0 = p[..., 0]
                x1 = p[..., 1]
                x2 = p[..., 2]                
            return eval(self.is_neumannbd)

    @cartesian
    def is_dirichlet_boundary(self, p):
        if self.is_dirichletbd is None:
            shape = p.shape[:-1]
            return np.zeros(shape,dtype = np.bool_)
        else:
            dim = self.dim
            if dim == 2:
                x0 = p[..., 0]
                x1 = p[..., 1]
            elif dim == 3:
                x0 = p[..., 0]
                x1 = p[..., 1]
                x2 = p[..., 2] 
            return eval(self.is_dirichletbd)

    def stress_direction(self,stress_direction=None):
        stress_direction = self.stress_directions if stress_direction is None else stress_direction
        return stress_direction


    
    def compliance_tensor(self, phi):
        lam = self.lam
        mu = self.mu
        dim = self.dim
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:dim], axis=-1)
        aphi[..., 0:dim] -= lam/(2*mu+dim*lam)*t[..., np.newaxis]
        aphi /= 2*mu
        return aphi    


    def Numpyfunction(self,f,p):
        dim = self.dim
        shape = p.shape #(...,dim)
        if dim != shape[-1]:
            print('dimension is not right!')
            return
        else:
            sym_val = f
            val_dim = sym_val.shape
            x = self.x
            if len(val_dim) == 1:  
                shape = shape[:-1]
                shape+=val_dim
                val = np.zeros(shape,dtype=float)                             
                for i in range(val_dim[0]):
                        idx = False
                        for k in range(dim):
                            idx = (idx or (str(x[k]) in str(sym_val[i])))
                        if  idx:
                            lam_val = sp.lambdify(self.x,sym_val[i],'numpy')
                            if dim == 2:
                                val[...,i] = lam_val(p[...,0],p[...,1])
                            elif dim == 3:
                                val[...,i] = lam_val(p[...,0],p[...,1],p[...,2])
                        else:
                            #print(sym_val[i][j])
                            val[...,i]+=float(sym_val[i])
                return val
            elif len(val_dim) == 2:
                shape = shape[:-1]
                shape +=val_dim
                val = np.zeros(shape,dtype=float)   
                for i in range(val_dim[0]):
                    for j in range(val_dim[1]):
                        idx = False
                        for k in range(dim):
                            idx = (idx or (str(x[k]) in str(sym_val[i][j])))
                        if  idx:
                            lam_val = sp.lambdify(self.x,sym_val[i][j],'numpy')
                            if dim == 2:
                                val[...,i,j] = lam_val(p[...,0],p[...,1])
                            elif dim == 3:
                                val[...,i,j] = lam_val(p[...,0],p[...,1],p[...,2])
                        else:
                            #print(sym_val[i][j])
                            val[...,i,j]+=float(sym_val[i][j])
                return val
        

if __name__ == '__main__':
    import sympy as sp
    import numpy as np

    x = sp.symbols('x0:2')
    #u = [sp.sin(sp.pi*x[0]),sp.sin(sp.pi*x[1]),sp.sin(sp.pi*x[2])]
    pi = sp.pi
    sin = sp.sin
    exp = sp.exp
    #u = [pi/2*sin(pi*x[0])**2*sin(2*pi*x[1]),-pi/2*sin(pi*x[1])**2*sin(2*pi*x[0])]
    #pdes = HuangModel2d(lam=1,mu=0.5)

    u = [exp(x[0]-x[1])*x[0]*(1-x[0])*x[1]*(1-x[1]),sin(pi*x[0])*sin(pi*x[1])]
    #u = [1.2,x[1]**2]
    #pdes = Model2d(lam=1,mu=0.5)

    pde = GenLinearElasticitymodel(u,x,Neumannbd='x0==1.0')
    pdes = GenLinearElasticitymodels(u,x,Neumannbd='x0==1.0')


    #p = np.random.random((5,10,3))

    p = np.random.random((200,2))
    #print(pdes.displacement(p).shape)
    #print(np.max(pde.displacement(p)[...,1]-pdes.displacement(p)[...,1]))

    print(np.max(pde.grad_displacement(p)-pdes.grad_displacement(p)))
    print(np.max(pde.stress(p)-pdes.stress(p)))

    print(np.max(pde.div_stress(p)-pdes.div_stress(p)))

    print(np.max(pde.source(p)-pdes.source(p)))

    print(np.max(pde.dirichlet(p)-pdes.dirichlet(p)))
