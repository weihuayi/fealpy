from fealpy.cem import EMFDTDSim
from fealpy.backend import backend_manager as bm
from fealpy.mesh import  UniformMesh2d, UniformMesh3d
import pandas as pd
import pytest

class TestEMFDTDSim:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])

    def test_cell_location(self, backend):
        """
        Test the correctness of the function "output the corresponding unit of the specified point"
        """
        domain = [0, 1, 0, 1] 
        h=0.1
        NX=int((domain[1]-domain[0])/h)
        NY=int((domain[3]-domain[2])/h)
        mesh2d = UniformMesh2d((0, NX, 0, NY), h=(h, h), origin=(domain[0], domain[2]))
        em1=EMFDTDSim(mesh2d,R=0.5,permittivity=1,permeability=1)

        a1=bm.array([0.15,0.25])
        b1=bm.array([0.28,0.76])

        assert em1.cell_location(a1)==(1,2)
        assert em1.cell_location(b1)==(2,7)

        domain2 = [0, 1, 0, 1, 0, 1] 
        h=0.1
        NX = int((domain2[1]-domain2[0])/h)
        NY = int((domain2[3]-domain2[2])/h)
        NZ = int((domain2[5]-domain2[4])/h)
        mesh3d = UniformMesh3d((0, NX, 0, NY, 0, NZ), h=(h, h, h), origin=(domain2[0], domain2[2], domain2[4]))
        em1=EMFDTDSim(mesh3d,R=0.5,permittivity=1,permeability=1)

        a2=bm.array([0.15, 0.25, 0.2])
        b2=bm.array([0.28, 0.76, 0.7])

        assert em1.cell_location(a2)==(1,2,2)
        assert em1.cell_location(b2)==(2,7,7)
    

    def test_node_location(self, backend):
        """
        Test the correctness of the function "Locate the nearest grid nodes for a set of points in a uniform mesh."
        """
        
        domain = [0, 1, 0, 1] 
        h=0.1
        NX=int((domain[1]-domain[0])/h)
        NY=int((domain[3]-domain[2])/h)
        mesh2d = UniformMesh2d((0, NX, 0, NY), h=(h, h), origin=(domain[0], domain[2]))
        em1=EMFDTDSim(mesh2d,R=0.5,permittivity=1,permeability=1)

        a=bm.array([0.16,0.26])
        b=bm.array([0.28,0.76])

        assert em1.node_location(a)==(2,3)
        assert em1.node_location(b)==(3,8)

        domain2 = [0, 1, 0, 1, 0, 1] 
        h=0.1
        NX = int((domain2[1]-domain2[0])/h)
        NY = int((domain2[3]-domain2[2])/h)
        NZ = int((domain2[5]-domain2[4])/h)
        mesh3d = UniformMesh3d((0, NX, 0, NY, 0, NZ), h=(h, h, h), origin=(domain2[0], domain2[2], domain2[4]))
        em1=EMFDTDSim(mesh3d,R=0.5,permittivity=1,permeability=1)

        a2=bm.array([0.15, 0.25, 0.2])
        b2=bm.array([0.28, 0.76, 0.7])

        assert em1.node_location(a2)==(2,3,2)
        assert em1.node_location(b2)==(3,8,7)

    # def test_
    bm.set_backend('numpy')

    def test_2d_convergence(self, backend):
        """
        Testing the second order convergence of two dimensional difference schemes
        """
        h=0.0625

        domain = [0, 1, 0, 1] 

        NX=int((domain[1]-domain[0])/h)
        NY=int((domain[3]-domain[2])/h)

        device = 'cpu'

        mesh = UniformMesh2d((0, NX, 0, NY), h=(h, h), origin=(domain[0], domain[2]),ftype = bm.float64,device=device) # 建立结构网格对象

        time=800
        m=1
        n=1
        E0=1.0
        c = 299792458.0
        mu0: float = 4e-7 * bm.pi
        omega = c * bm.pi * bm.sqrt(bm.array((m)**2 + (n)**2))

        dt = bm.sqrt(bm.array(2))*0.0625/(128*c)  

        em1=EMFDTDSim(mesh,permittivity=1,permeability=1,dt=dt) #模拟器的初始化

        def Ez_func(X, Y, t):
            return E0 * bm.sin(m*bm.pi*X) * bm.sin(n*bm.pi*Y)* bm.cos(omega*t)

        def Hx_func(X, Y, t):
            return -E0/(mu0*omega) * (n*bm.pi) * bm.sin(m*bm.pi*X) * bm.cos(n*bm.pi*Y) * bm.sin(omega*t)

        def Hy_func(X, Y, t):
            return E0/(mu0*omega) * (m*bm.pi) * bm.cos(m*bm.pi*X) * bm.sin(n*bm.pi*Y) * bm.sin(omega*t)

        L2 = []
        H1 = []

        maxit = 5
        for i in range(maxit):

            em1.initialize_field('E_z', Ez_func,time)
            em1.initialize_field('H_x', Hx_func,time)
            em1.initialize_field('H_y', Hy_func,time)

            em1.run(time=time)  

            h = mesh.h[0]  
            eh1 = em1.E['z'][-1] - em1.true_solutions['E_z'][-1]
            eh2 = em1.H['x'][-1] - em1.true_solutions['H_x'][-1]
            eh3 = em1.H['y'][-1] - em1.true_solutions['H_y'][-1]

            eu11 = bm.sum(eh1**2) * (h**2)  
            eu12 = bm.sum(eh2**2) * (h**2)  
            eu13 = bm.sum(eh3**2) * (h**2)  
            eu1 = eu11+eu12+eu13

            dx1 = eh1[1:  , :] - eh1[:-1, :]  
            dy1 = eh1[ :  ,1:] - eh1[ :  ,:-1] 
            dx2 = eh2[1:  , :] - eh2[:-1, :]  
            dy2 = eh2[ :  ,1:] - eh2[ :  ,:-1] 
            dx3 = eh3[1:  , :] - eh3[:-1, :]  
            dy3 = eh3[ :  ,1:] - eh3[ :  ,:-1] 

            eu21 = (bm.sum(dx1**2) + bm.sum(dy1**2))
            eu22 = (bm.sum(dx2**2) + bm.sum(dy2**2))
            eu23 = (bm.sum(dx3**2) + bm.sum(dy3**2))

            eu2 = eu21+eu22+eu23
            # H1 范数
            eu = bm.sqrt(eu1+eu2)

            if i==0:
                L2.append({"N": 1/mesh.h[0],"L2":bm.sqrt(eu1)})
                H1.append({"N": 1/mesh.h[0],"H1":eu})

            else:

                ln2 = bm.log(2)
                eL2_now = float(bm.sqrt(eu1))
                eL2_prev = L2[-1]["L2"]
                eH1_now = float(eu)
                eH1_prev = H1[-1]["H1"]
                r1 = bm.log(eL2_now / eL2_prev) / -ln2
                r2 = bm.log(eH1_now / eH1_prev) / -ln2
                L2.append({"N": 1/mesh.h[0],"L2":bm.sqrt(eu1),"R":r1})
                H1.append({"N": 1/mesh.h[0],"H1":eu,"R":r2})

            if i < maxit:
                mesh.uniform_refine()
                # h= h/2
                # NX=int((domain[1]-domain[0])/h)
                # NY=int((domain[3]-domain[2])/h)
                # mesh = UniformMesh2d((0, NX, 0, NY), h=(h, h), origin=(domain[0], domain[2]),ftype = bm.float64,device=device) 
                em1=EMFDTDSim(mesh,permittivity=1,permeability=1,dt=dt)
            
        # 打印结果
        df1=pd.DataFrame(L2)
        df2=pd.DataFrame(H1)
        df1["L2"] = df1["L2"].apply(lambda x: f"{x:.4g}")
        df2["H1"] = df2["H1"].apply(lambda x: f"{x:.4g}")
        print(df1)
        print(df2)

    def test_3d_convergence(self, backend):
        """
        Testing the second order convergence of three dimensional difference schemes
        """
        domain = [0, 1, 0, 1,0, 1] 
        h = 0.0625

        NX=int((domain[1]-domain[0])/h)
        NY=int((domain[3]-domain[2])/h)
        NZ=int((domain[5]-domain[4])/h)

        mesh = UniformMesh3d((0, NX, 0, NY,0, NZ), h=(h, h, h), origin=(domain[0], domain[2], domain[4])) # 建立结构网格对象

        time=400
        m, n, p = 1, 1, 1
        E0 = 1.0
        c = 299_792_458.0
        mu0 = 4e-7 * bm.pi
        eps0 = 1.0 / (mu0 * c**2)

        # 波数和角频率
        kx = m * bm.pi
        ky = n * bm.pi
        kz = p * bm.pi
        kc2 = kx**2 + ky**2
        omega = c * bm.sqrt(bm.asarray(kc2 + kz**2))

        def Ex_func(X, Y, Z, t):
            return (
                -E0 * (kx * kz) / kc2
                * bm.cos(kx * X)
                * bm.sin(ky * Y)
                * bm.sin(kz * Z)
                * bm.cos(omega * t)
            )

        def Ey_func(X, Y, Z, t):
            return (
                -E0 * (ky * kz) / kc2
                * bm.sin(kx * X)
                * bm.cos(ky * Y)
                * bm.sin(kz * Z)
                * bm.cos(omega * t)
            )

        def Ez_func(X, Y, Z, t):
            return (
                E0
                * bm.sin(kx * X)
                * bm.sin(ky * Y)
                * bm.cos(kz * Z)
                * bm.cos(omega * t)
            )

        def Hx_func(X, Y, Z, t):
            return (
                -E0 * (omega * eps0 * ky) / kc2
                * bm.sin(kx * X)
                * bm.cos(ky * Y)
                * bm.cos(kz * Z)
                * bm.sin(omega * t)
            )

        def Hy_func(X, Y, Z, t):
            return (
                E0 * (omega * eps0 * kx) / kc2
                * bm.cos(kx * X)
                * bm.sin(ky * Y)
                * bm.cos(kz * Z)
                * bm.sin(omega * t)
            )

        def Hz_func(X, Y, Z, t):

            return bm.zeros_like(X)

        dt=bm.sqrt(bm.asarray(2))*0.0625/(128*c)

        em1=EMFDTDSim(mesh,permittivity=1,permeability=1,dt=dt) #模拟器的初始化

        L2 = []
        H1 = []

        maxit = 4
        for i in range(maxit):


            em1.initialize_field('E_z', Ez_func,time)

            em1.initialize_field('E_x', Ex_func,time)
            em1.initialize_field('E_y', Ey_func,time)
            em1.initialize_field('H_x', Hx_func,time)
            em1.initialize_field('H_y', Hy_func,time)
            em1.initialize_field('H_z', Hz_func,time)
            
            em1.run(time=time)  

            h = mesh.h[0]  
            
            eh1 = em1.E['x'][-1] - em1.true_solutions['E_x'][-1] 
            eh2 = em1.E['y'][-1] - em1.true_solutions['E_y'][-1]
            eh3 = em1.E['z'][-1] - em1.true_solutions['E_z'][-1]
            eh4 = em1.H['x'][-1] - em1.true_solutions['H_x'][-1]
            eh5 = em1.H['y'][-1] - em1.true_solutions['H_y'][-1]
            eh6 = em1.H['z'][-1] - em1.true_solutions['H_z'][-1]

            eu11 = bm.sum(eh1**2) * (h**3)
            eu12 = bm.sum(eh2**2) * (h**3)  
            eu13 = bm.sum(eh3**2) * (h**3)  
            eu14 = bm.sum(eh4**2) * (h**3)  
            eu15 = bm.sum(eh5**2) * (h**3)  
            eu16 = bm.sum(eh6**2) * (h**3)   

            eu1 = eu11+eu12+eu13+eu14+eu15+eu16

            dx1 = eh1[1:, :, :] - eh1[:-1, :, :]
            dy1 = eh1[:, 1:, :] - eh1[:, :-1, :]
            dz1 = eh1[:, :, 1:] - eh1[:, :, :-1]

            dx2 = eh2[1:, :, :] - eh2[:-1, :, :]
            dy2 = eh2[:, 1:, :] - eh2[:, :-1, :]
            dz2 = eh2[:, :, 1:] - eh2[:, :, :-1]

            dx3 = eh3[1:, :, :] - eh3[:-1, :, :]
            dy3 = eh3[:, 1:, :] - eh3[:, :-1, :]
            dz3 = eh3[:, :, 1:] - eh3[:, :, :-1]

            dx4 = eh4[1:, :, :] - eh4[:-1, :, :]
            dy4 = eh4[:, 1:, :] - eh4[:, :-1, :]
            dz4 = eh4[:, :, 1:] - eh4[:, :, :-1]

            dx5 = eh5[1:, :, :] - eh5[:-1, :, :]
            dy5 = eh5[:, 1:, :] - eh5[:, :-1, :]
            dz5 = eh5[:, :, 1:] - eh5[:, :, :-1]

            dx6 = eh6[1:, :, :] - eh6[:-1, :, :]
            dy6 = eh6[:, 1:, :] - eh6[:, :-1, :]
            dz6 = eh6[:, :, 1:] - eh6[:, :, :-1]


            eu21 = (bm.sum(dx1**2) + bm.sum(dy1**2) + bm.sum(dz1**2)) * h
            eu22 = (bm.sum(dx2**2) + bm.sum(dy2**2) + bm.sum(dz2**2)) * h
            eu23 = (bm.sum(dx3**2) + bm.sum(dy3**2) + bm.sum(dz3**2)) * h
            eu24 = (bm.sum(dx4**2) + bm.sum(dy4**2) + bm.sum(dz4**2)) * h
            eu25 = (bm.sum(dx5**2) + bm.sum(dy5**2) + bm.sum(dz5**2)) * h
            eu26 = (bm.sum(dx6**2) + bm.sum(dy6**2) + bm.sum(dz6**2)) * h
            eu2 = eu21+eu22+eu23+eu24+eu25+eu26

            eu = bm.sqrt(eu1 + eu2)


            if i==0:
                L2.append({"N": 1/mesh.h[0],"L2":bm.sqrt(eu1)})
                H1.append({"N": 1/mesh.h[0],"H1":eu})
            else:

                ln2 = bm.log(bm.asarray(2))
                eL2_now = float(bm.sqrt(eu1))
                eL2_prev = L2[-1]["L2"]
                eH1_now = float(eu)
                eH1_prev = H1[-1]["H1"]
                r1 = bm.log(eL2_now / eL2_prev) / -ln2
                r2 = bm.log(eH1_now / eH1_prev) / -ln2
                L2.append({"N": 1/mesh.h[0],"L2":bm.sqrt(eu1),"R":r1})
                H1.append({"N": 1/mesh.h[0],"H1":eu,"R":r2})


            if i < maxit:
                mesh.uniform_refine()
                em1=EMFDTDSim(mesh,permittivity=1,permeability=1,dt=dt)

        df1=pd.DataFrame(L2)
        df2=pd.DataFrame(H1)

        df1["L2"] = df1["L2"].apply(lambda x: f"{x:.4g}")
        df2["H1"] = df2["H1"].apply(lambda x: f"{x:.4g}")

        print(df1)
        print(df2)
