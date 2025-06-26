from fealpy.backend import backend_manager as bm
from scipy.sparse import coo_matrix
from fealpy.sparse import COOTensor as cm


class NSFVMSolver:
    def __init__(self, pde, ht):

        self.pde = pde
        self.mesh = pde.mesh
        self.mesh0 = pde.mesh0
        self.mesh1 = pde.mesh1
        self.h = 1/pde.nx
        self.ht = ht
        self.NC = pde.mesh.number_of_cells()
        self.NC0 = pde.mesh0.number_of_cells()
        self.NC1 = pde.mesh1.number_of_cells()
        self.edge = pde.mesh.entity('edge')
        self.node = pde.mesh.entity('node')
        self.c2c = pde.mesh.cell_to_cell()
        self.c2n0 = pde.mesh0.cell_to_node()
        self.c2c0 = pde.mesh0.cell_to_cell()
        self.c2n1 = pde.mesh1.cell_to_node()
        self.c2c1 = pde.mesh1.cell_to_cell()

        self.cell2edge = pde.mesh.cell_to_edge()
        cell2edge = self.cell2edge
        EN_u = bm.array(list(set(cell2edge[:, 3]) | set(cell2edge[:, 1])))
        EN_u = bm.sort(EN_u)
        self.EN_u = bm.unique(EN_u)
        EN_v = bm.array(list(set(cell2edge[:, 0]) | set(cell2edge[:, 2])))
        EN_v = bm.sort(EN_v)
        self.EN_v = bm.unique(EN_v)
        self.point = pde.mesh.entity_barycenter('edge')
        self.point_u = self.point[self.EN_u]
        self.point_v = self.point[self.EN_v]
        self.point_p = self.pde.mesh.entity_barycenter('cell')
        self.u_nC0 = self.pde.velocity_u(self.point_u)
        self.v_nC0 = pde.velocity_v(self.point_v)

        self.flag = self.c2c[bm.arange(self.NC)] == bm.arange(self.NC)[:, None] # 判断是边界的
        self.flag0 = self.c2c0[bm.arange(self.NC0)] == bm.arange(self.NC0)[:, None]#判断在u网格系统中是边界的
        self.flag1 = self.c2c1[bm.arange(self.NC0)] == bm.arange(self.NC0)[:, None]#判断在v网格系统中是边界的     
       
    def NS_Lform_us(self):

        h = self.h
        ht = self.ht
        NC0 = self.NC0
        c2c0 = self.c2c0
        flag0 = self.pde.c2c0[bm.arange(NC0)] == bm.arange(NC0)[:, None]
        flag_u1 = self.pde.is_boundary(self.point_u)
        bE_u1 = self.point_u[flag_u1]
        index_u1 = [bm.where(bm.all(self.point_u == bE, axis=1))[0][0] for bE in bE_u1]
        index_u1 = bm.asarray(index_u1)


        I = bm.arange(NC0)
        J = bm.arange(NC0)
        indices = bm.stack([I, J])
        val = (h**2/ht+1/2*h/h*4)*bm.ones(NC0)
        A_us = cm(indices, val, (NC0, NC0))

        I = bm.where(~flag0)[0]
        J = c2c0[~flag0]
        indices = bm.stack([I, J])
        val = -1/2*bm.ones(I.shape)
        A_us += cm(indices, val, (NC0, NC0))

        I = bm.where(flag0)[0]
        J = bm.where(flag0)[0]
        indices = bm.stack([I, J])
        val = 1/2*bm.ones(I.shape)
        A_us += cm(indices, val, (NC0, NC0))

        I, J = A_us._indices
        val = A_us._values
        I = I.int()
        index_u1 = index_u1.int()
        Amask = ~bm.isin(I, index_u1) 

        I = I[Amask]
        J = J[Amask]
        indices = bm.stack([I, J])
        val = val[Amask]
        A_u0 = cm(indices, val, (NC0, NC0))

        I = index_u1
        J = index_u1
        indices = bm.stack([I, J])
        val = bm.ones(len(I))
        A_u0 += cm(indices, val, (NC0, NC0))
        A_u0 = A_u0.tocoo()
        A_u0 = A_u0.astype(bm.float64)
        return A_u0

    def NS_Lform_vs(self):

        h = self.h
        ht = self.ht
        NC1 = self.pde.mesh1.number_of_cells()
        c2c1 = self.c2c1
        flag1 = c2c1[bm.arange(NC1)] == bm.arange(NC1)[:, None]
        flag_v1 = self.pde.is_boundary(self.point_v)
        bE_v1 = self.point_v[flag_v1]
        index_v1 = [bm.where(bm.all(self.point_v == bE, axis=1))[0][0] for bE in bE_v1]
        index_v1 = bm.asarray(index_v1)

        I = bm.arange(NC1)
        J = bm.arange(NC1)
        indices = bm.stack([I, J])
        val = (h**2/ht+2)*bm.ones(NC1)
        A_vs = cm(indices, val, (NC1, NC1))

        I = bm.where(~flag1)[0] 
        J = c2c1[~flag1]
        indices = bm.stack([I, J])
        val = -1/2*bm.ones(I.shape)
        A_vs += cm(indices, val, (NC1, NC1))

        I = bm.where(flag1)[0] 
        J = bm.where(flag1)[0]
        indices = bm.stack([I, J])
        val = 1/2*bm.ones(I.shape)
        A_vs += cm(indices, val, (NC1, NC1))

        I, J = A_vs._indices
        val = A_vs._values
        I = I.int()
        index_v1 = index_v1.int()
        Amask = ~bm.isin(I, index_v1)
        I = I[Amask]
        J = J[Amask]
        indices = bm.stack([I, J])
        val = val[Amask]
        A_v0 = cm(indices, val, (NC1, NC1))

        I = index_v1
        J = index_v1
        indices = bm.stack([I, J])
        val = bm.ones(len(I))
        A_v0 += cm(indices, val, (NC1, NC1))
        A_v0 = A_v0.tocoo()
        A_v0 = A_v0.astype(bm.float64)
        return A_v0

    def NS_Bform_us(self, u0, v0, p0):
        h = self.h
        ht = self.ht
        NC0 = self.NC0
        c2c0 = self.c2c0
        c2c1 = self.c2c1
        flag0 = self.flag0
        flag1 = self.flag1
        point_u = self.point_u
        u0 = bm.tensor(u0)
        u0 = u0.double()
        v0 = bm.tensor(v0)
        v0 = v0.double()
        p0 = bm.tensor(p0)
        p0 = p0.double()
        
        b_u0 = bm.zeros(NC0)
        uE = u0[c2c0[..., 1]]
        uW = u0[c2c0[..., 3]]
        uN = u0[c2c0[..., 2]]
        uS = u0[c2c0[..., 0]]
        uE[flag0[..., 1]] = 0
        uW[flag0[..., 3]] = 0
        uN[flag0[..., 2]] = 0
        uS[flag0[..., 0]] = 0
        
        ue = u0 + uE
        ue[flag0[..., 1]] = 0
        uw = u0 + uW
        uw[flag0[..., 3]] = 0
        un = u0 + uN
        un[flag0[..., 2]] = 0
        us = u0 + uS
        us[flag0[..., 0]] = 0
        vn = bm.zeros(NC0).double()
        vs = bm.zeros(NC0).double()
        vn[~flag0[..., 1]] = v0[~flag1[..., 0]] + v0[c2c1[..., 3]][~flag1[..., 0]]
        vn[flag0[..., 2]] = 0
        vs[~flag0[..., 3]] = v0[~flag1[..., 2]] + v0[c2c1[..., 1]][~flag1[..., 2]]
        vs[flag0[..., 0]] = 0
        
        a_n0 = bm.ones(NC0)*(h**2/ht-2) 
        a_n0 += (1/4)*(uw - ue + vs - vn)*h
        b_uC = a_n0 * u0

        a_ne = bm.ones(NC0)*(h/h)/2
        a_ne -= (1/4)*(ue)*h
        a_nw = bm.ones(NC0)*(h/h)/2
        a_nw -= (1/4)*(-uw)*h
        a_nn = bm.ones(NC0)*(h/h)/2
        a_nn -= (1/4)*(vn)*h
        a_ns = bm.ones(NC0)*(h/h)/2
        a_ns -= (1/4)*(-vs)*h
        b_ue = a_ne * uE
        b_uw = a_nw * uW
        b_un = a_nn * uN
        b_us = a_ns * uS
        b_uF = b_ue + b_uw + b_un + b_us
        
        Q_uC = bm.zeros(NC0)
        Q_uC = Q_uC.double()
        Q_uC[~flag0[..., 1]] = -(p0 - p0[self.c2c[..., 3]])*h
        #Q_uC += self.pde.source_u(point_u)*h**2
        b_u0 += b_uC + b_uF + Q_uC
        b_u0 = b_u0.double()
        
        return b_u0 
    
    def NS_Bform_vs(self, u0, v0, p0):
        h = self.h
        ht = self.ht
        NC1 = self.NC1
        c2c0 = self.c2c0
        c2c1 = self.c2c1
        flag0 = self.flag0
        flag1 = self.flag1
        point_v = self.point_v
        u0 = bm.tensor(u0)
        u0 = u0.double()
        v0 = bm.tensor(v0)
        v0 = v0.double()
        p0 = bm.tensor(p0)
        p0 = p0.double()

        b_v0 = bm.zeros(NC1)
        vE = v0[c2c1[..., 1]]
        vW = v0[c2c1[..., 3]]
        vN = v0[c2c1[..., 2]]
        vS = v0[c2c1[..., 0]]
        vE[flag1[..., 1]] = 0
        vW[flag1[..., 3]] = 0
        vN[flag1[..., 2]] = 0
        vS[flag1[..., 0]] = 0
        
        ue = bm.zeros(NC1).double()
        uw = bm.zeros(NC1).double()
        #vn = bm.zeros(NC1)
        #vs = bm.zeros(NC1)
        ue[~flag1[..., 0]] = u0[~flag0[..., 3]] + u0[c2c0[..., 2]][~flag0[..., 3]]
        ue[flag1[..., 3]] = 0
        uw[~flag1[..., 2]] = u0[~flag0[..., 1]] + u0[c2c1[..., 0]][~flag0[..., 1]]
        uw[flag1[..., 1]] = 0
        ve = v0 + vE
        ve[flag1[..., 1]] = 0
        vw = v0 + vW
        vw[flag1[..., 3]] = 0
        vn = v0 + v0[c2c1[..., 2]]
        vn[flag1[..., 2]] = 0
        vs = v0 + v0[c2c1[..., 0]]
        vs[flag1[..., 0]] = 0

        a_nC1 = bm.ones(NC1)*(h**2/ht-2) 
        a_nC1 += (1/4)*(uw - ue + vs - vn)*h
        b_vC = a_nC1 * v0

        a_ne = bm.ones(NC1)*(h/h)/2
        a_ne -= (1/4)*(ue)*h
        a_nw = bm.ones(NC1)*(h/h)/2
        a_nw -= (1/4)*(-uw)*h
        a_nn = bm.ones(NC1)*(h/h)/2
        a_nn -= (1/4)*(vn)*h
        a_ns = bm.ones(NC1)*(h/h)/2
        a_ns -= (1/4)*(-vs)*h
        b_ve = a_ne * vE
        b_vw = a_nw * vW
        b_vn = a_nn * vN
        b_vs = a_ns * vS
        b_vF = b_ve + b_vw + b_vn + b_vs

        Q_vC = bm.zeros(NC1)
        Q_vC = Q_vC.double()
        Q_vC[~flag1[..., 2]] = -(p0 - p0[self.c2c[..., 0]])*h
        #Q_vC += self.pde.source_v(point_v)*h**2
        b_v0 += b_vC + b_vF + Q_vC
        b_v0 = b_v0.double()

        return b_v0

    def DirichletBC_1(self, u0, v0, p0):
        h = self.h
        mesh = self.pde.mesh
        pde = self.pde
        c2n0 = self.c2n0
        c2n1 = self.c2n1
        flag0 = self.flag0
        flag1 = self.flag1
        u_nC0 = self.u_nC0
        v_nC0 = self.v_nC0
        node = mesh.entity('node')
        point_node = mesh.entity_barycenter('node')
       
        b_u0 = self.NS_Bform_us(u0, v0, p0)
        bu = bm.zeros(self.NC0)
        bunode0 = point_node[c2n0[..., 0][flag0[..., 0]]]
        bunode1 = point_node[c2n0[..., 3][flag0[..., 2]]]
        bu[flag0[..., 0]] = pde.u_dirichlet(bunode0) * pde.v_dirichlet(bunode0)
        bu[flag0[..., 2]] = pde.u_dirichlet(bunode1) * pde.v_dirichlet(bunode1)

        data0 = h*bu
        data0[flag0[..., 0]] += (2*h/h)*pde.u_dirichlet(bunode0) 
        data0[flag0[..., 0]] += -1/2*(h/h)*u_nC0[flag0[..., 0]]
        data0[flag0[..., 2]] += (2*h/h)*pde.u_dirichlet(bunode1) 
        data0[flag0[..., 2]] += -1/2*(h/h)*u_nC0[flag0[..., 2]]
        b_u0 += data0
        b_u0[flag0[..., 1]] = u_nC0[flag0[..., 1]].to(dtype=b_u0.dtype)
        b_u0[flag0[..., 3]] = u_nC0[flag0[..., 3]].to(dtype=b_u0.dtype)

        b_v0 = self.NS_Bform_vs(u0, v0, p0) 
        bv = bm.zeros(self.NC1)
        bvnode0 = point_node[c2n1[..., 0][flag1[..., 3]]]
        nd = pde.nx+1
        bvnode1 = node[-nd:]
        bv[flag1[..., 3]] = pde.u_dirichlet(bvnode0) * pde.v_dirichlet(bvnode0)
        bv[flag1[..., 1]] = pde.u_dirichlet(bvnode1) * pde.v_dirichlet(bvnode1)
        data1 = h*bv
        data1[flag1[..., 3]] += (2*h/h)*self.pde.v_dirichlet(bvnode0) - 1/2*(h/h)*v_nC0[flag1[..., 3]]
        data1[flag1[..., 1]] += (2*h/h)*self.pde.v_dirichlet(bvnode1) - 1/2*(h/h)*v_nC0[flag1[..., 1]]       
        b_v0 += data1
        b_v0[flag1[..., 0]] = v_nC0[flag1[..., 0]].to(dtype=b_u0.dtype)
        b_v0[flag1[..., 2]] = v_nC0[flag1[..., 2]].to(dtype=b_u0.dtype)

        return b_u0, b_v0
    
    def NS_Lform_p(self):
        h = self.h
        NC = self.NC
        c2c = self.c2c
        flag = self.flag

        I = bm.arange(NC)
        J = bm.arange(NC)
        indices = bm.stack([I, J])
        val = 4*h/h*bm.ones(NC)
        A_p = cm(indices, val, (NC, NC))

        I = bm.where(flag)[0] 
        J = bm.where(flag)[0]
        indices = bm.stack([I, J])
        val = (h/(h/2)-h/h)*bm.ones(I.shape)
        A_p += cm(indices, val, (NC, NC))

        I = bm.where(~flag)[0] 
        J = c2c[~flag]
        indices = bm.stack([I, J])
        val = -h/h*bm.ones(I.shape)
        A_p += cm(indices, val, (NC, NC))
        A_p = A_p.tocoo()
        A_p = A_p.astype(bm.float64)

        return A_p
    
    def NS_Bform_p(self, us, vs, p0):
        
        NC = self.NC
        flag0 = self.flag0
        flag1 = self.flag1

        b_p = bm.zeros(NC)
        A_p = self.NS_Lform_p()
        b_pP = A_p @ p0

        Q_p = us[~flag0[..., 3]] - us[~flag0[..., 1]]
        Q_p += vs[~flag1[..., 0]] - vs[~flag1[..., 2]]
        Q_p = -self.h/self.ht * Q_p
        b_p = Q_p + b_pP
        b_p = b_p.double()

        return b_p

    def NS_Lform_u1(self):
        NC0 = self.NC0 

        I = bm.arange(NC0)
        J = bm.arange(NC0)
        indices = bm.stack([I, J])
        val = bm.ones(NC0)
        A_u1 = cm(indices, val, (NC0, NC0))
        A_u1 = A_u1.tocoo()
        A_u1 = A_u1.astype(bm.float64)

        return A_u1
    
    def NS_Lform_v1(self):
        NC1 = self.NC1

        I = bm.arange(NC1)
        J = bm.arange(NC1)
        indices = bm.stack([I, J])
        val = bm.ones(NC1)
        A_v1 = cm(indices, val, (NC1, NC1))
        A_v1 = A_v1.tocoo()
        A_v1 = A_v1.astype(bm.float64)

        return A_v1
    
    def NS_Bform_u1(self, us, p0, p1):
        h = self.h
        ht = self.ht
        c2c = self.c2c
        flag0 = self.flag0
        us = bm.tensor(us)
        us = us.double()

        b_u1 = us
        b_u1 = b_u1.double()
        b_u1[~flag0[..., 1]] -= ht/h * (p0[c2c[..., 3]] - p0)
        b_u1[~flag0[..., 1]] += ht/h * (p1[c2c[..., 3]] - p1)
        b_u1[flag0[..., 3]] = us[flag0[..., 3]]
        b_u1 = b_u1.double()

        return b_u1

    def NS_Bform_v1(self, vs, p0, p1):
        h = self.h
        ht = self.ht
        c2c = self.c2c
        flag1 = self.flag1
        vs = bm.tensor(vs)
        vs = vs.double()

        b_v1 = vs
        b_v1 = b_v1.double()
        b_v1[~flag1[..., 2]] -= ht/h * (p0[c2c[..., 0]] - p0)
        b_v1[~flag1[..., 2]] += ht/h * (p1[c2c[..., 0]] - p1)
        b_v1[flag1[..., 0]] = vs[flag1[..., 0]]
        b_v1 = b_v1.double()
        return b_v1
