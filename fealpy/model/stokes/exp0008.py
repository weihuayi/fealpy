from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d
from typing import Sequence


class Exp0008(BoxMesher2d):
    """
    Analytic solution to the 2D Stokes equations:

        -μ Δu + ∇p = f  in Ω = [0,1]^2
        ∇·u = 0         in Ω
        u = g           on ∂Ω

        f =(0, 0)

    """

    def __init__(self, option: dict = {}):
        self.box = [0, 1, 0, 1]
        self.mu = bm.tensor(option.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def viscosity(self) -> float:
        return self.mu

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        f = bm.zeros_like(p)
        return f

    def init_mesh(self, node=None, cell=None):
        from fealpy.mesh import TriangleMesh
        if node is None:
            node = bm.array([
            [-0.5, -1],
            [0.5, -1],
            [0.5, 0],
            [1.5, 0],
            [1.5, 1],
            [-1.5, 1],
            [-1.5, 0],
            [-0.5, 0],
            [-0.5, -0.5],
            [0, -1],
            [0.5, -0.5],
            [1, 0],
            [0.5, 1],
            [-0.5, 1],
            [-1, 0],
            [-0.5955, 0],
            [-0.5, -0.0955],
            [-0.5, -0.9045],
            [-0.4045, -1],
            [0.4045, -1],
            [0.5, -0.9045],
            [0.5, -0.0955],
            [0.5955, 0],
            [1, 0.5],
            [0.5, 0.5],
            [0, 0.5],
            [-0.5, 0.5],
            [-1, 0.5],
            [-0.5955, 0.0955],
            [-0.5, 0.0955],
            [-0.4045, 0.0955],
            [-0.4045, 0],
            [-0.4045, -0.0955],
            [-0.4045, -0.9045],
            [0.4045, -0.9045],
            [0.4045, -0.0955],
            [0.4045, 0],
            [0.4045, 0.0955],
            [0.5, 0.0955],
            [0.5955, 0.0955],
            [0, -0.5],
            [0, 0]
                    ], dtype=bm.float64)
        if cell is None:
            cell = bm.array([
            [5, 6, 27],
            [6, 14, 27],
            [14, 28, 27],
            [14, 15, 28],
            [15, 7, 28],
            [7, 29, 28],
            [7, 30, 29],
            [7, 31, 30],
            [7, 32, 31],
            [7, 16, 32],
            [16, 8, 32],
            [8, 40, 32],
            [8, 33, 40],
            [8, 17, 33],
            [17, 0, 33],
            [0, 18, 33],
            [18, 9, 33],
            [9, 40, 33],
            [9, 34, 40],
            [9, 19, 34],
            [19, 1, 34],
            [1, 20, 34],
            [20, 10, 34],
            [10, 40, 34],
            [10, 35, 40],
            [10, 21, 35],
            [2, 35, 21],
            [2, 36, 35],
            [2, 37, 36],
            [2, 38, 37],
            [2, 39, 38],
            [2, 22, 39],
            [22, 11, 39],
            [11, 23, 39],
            [11, 3, 23],
            [3, 4, 23],
            [4, 12, 23],
            [12, 24, 23],
            [12, 25, 24],
            [12, 13, 25],
            [13, 26, 25],
            [13, 27, 26],
            [13, 5, 27],
            [26, 27, 28],
            [26, 28, 29],
            [26, 29, 30],
            [26, 30, 25],
            [25, 30, 41],
            [41, 30, 31],
            [41, 31, 32],
            [41, 32, 40],
            [41, 40, 35],
            [41, 35, 36],
            [41, 36, 37],
            [41, 37, 25],
            [25, 37, 24],
            [24, 37, 38],
            [38, 39, 24],
            [24, 39, 23]
                    ], dtype=bm.int64)

        return TriangleMesh(node, cell)




    @cartesian
    def is_inoutlet_boundary(self, p: TensorLike) -> TensorLike:                   
        """Check if point where velocity is defined is on boundary."""          
        x = p[..., 0]                                                           
        y = p[..., 1]                                                           
        atol = 1e-14                                                            
        # 检查是否接近 x=±1 或 y=±1                                             
        #on_boundary = (bm.abs(y- 0) <atol) & (x > -1 - atol) & (x < 1 + atol)
        flag1 = (bm.abs(x + 3/2) <= atol) & ( y >= 0 - atol) & (y<=1+atol) 
        flag2 = (bm.abs(x - 3/2) <= atol) & ( y >= 0 - atol) & (y<=1+atol) 
        on_boundary = flag1 | flag2
        return on_boundary

    #@cartesian
    #def is_wall_boundary(self, p: TensorLike) -> TensorLike:                   
    #    """Check if point where velocity is defined is on boundary."""          
    #    x = p[..., 0]                                                           
    #    y = p[..., 1]                                                           
    #    atol = 1e-14                                                            
    #    A = bm.array([0, -3])
    #    B1 = bm.array([-1, 0]) 
    #    B2 = bm.array([1, 0])
    #    PA = p - A[None, None, :]
    #    PB1 = p - B1[None, None, :]
    #    PB2 = p - B2[None, None, :]
    #    flag1 = bm.abs(bm.cross(PA, PB1)) <= eps                                 
    #    flag2 = bm.sum(PA*PB1,axis=-1 ) <= 0                                     
    #    flag3 = flag1 & flag2                                                    
    #    flag4 = bm.abs(bm.cross(PA, PB2)) <= eps                                 
    #    flag5 = bm.sum(PA*PB2,axis=-1 ) <= 0                                     
    #    flag6= flag4 & flag5                                                    
    #    on_boundary = flag3 | flag6
    #    return on_boundary

    @cartesian
    def velocity_boundary(self, p):
        x, y = p[..., 0], p[..., 1]
        eps = 1e-14 
        flag1 = self.is_inoutlet_boundary(p)
        u = bm.zeros_like(p, bm.float64)
        u[flag1, 0] = y[flag1]*(1 - y[flag1])
        return u

    @cartesian
    def grad_stream_function_boundary(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        g = -self.velocity_boundary(p)
        val = bm.einsum('eqd,ed->eq', g, t)
        return val 

    def order_edge(self, num):                                                      
        mesh = self.init_mesh() 
        node = mesh.entity('node')
        edge = mesh.entity('edge')                                                  
        enode = node[edge]
        menode = (enode[:, 0, :] + enode[:, 1, :])/2
        is_boundary_edge = mesh.boundary_edge_flag()                                
        edge = edge[is_boundary_edge]                                               
        dist = {}                                                                   
        i = 0                                                                       
        for u,v in edge:                                                            
            dist[u] = [u, v, i]                                                     
            i = i+ 1                                                                
        res = []                                                                    
        res.append(dist[num])                                                       
        for i in range(len(edge)-1):                                                
            temp = dist[dist[num][1]]                                               
            res.append(temp)                                                        
            num = temp[0]                                                           
        res = bm.array(res)                                                         
        bedge = res[..., :2]  # (BNE, 2, 2)                                           
        index = res[..., 2]   # (BNE,)                                              
        return bedge, index  

    @cartesian                                                                  
    def stream_function_boundary(self, p):                                                       
                                                                                    
        ## 判断点在第几个边界上                                                     
        value = bm.zeros_like(p.shape[:-1])                                         
        mesh = self.init_mesh() 
        node = mesh.entity('node')
        edge = mesh.entity('edge')                                                  
        enode = node[edge]
        menode = (enode[:, 0, :] + enode[:, 1, :])/2

        is_boundary_edge = mesh.boundary_edge_flag()                                
        bedge, index = self.order_edge(0)                                          
        bnode = mesh.entity('node')[bedge]                                          
        BNE = bedge.shape[0]                                                        
        k = bm.zeros(p.shape[:-1], dtype=bm.int32)                                  
        eps = 1e-14                                                                 
        for i in range(BNE):                                                        
            PA = p - bnode[i, 0, :]                                                 
            PB = p - bnode[i, 1, :]                                                 
            flag1 = bm.abs(bm.cross(PA, PB)) <= eps                                 
            flag2 = bm.sum(PA*PB,axis=-1 ) <= 0                                     
            flag = flag1 & flag2                                                    
            k[flag] = i                                                             
        # 计算边积分                                                                
        q = 10                                                                       
        qf = mesh.quadrature_formula(q, 'edge')                                     
        bcs, ws = qf.get_quadrature_points_and_weights()                            
        point = bm.bc_to_points(bcs, node, entity = bedge) #(BNE, NQ, GD)           
        ei = bm.zeros_like(bedge)                                                   
        en = mesh.edge_unit_normal()[is_boundary_edge][index]                       
        em = mesh.edge_length()[is_boundary_edge][index]                            
        ei = bm.einsum('eqd,q,e,ed->e', self.velocity_boundary(point),ws,em,en)                          
                                                                                    
        # 计算第一部分的积分                                                        
        eii = bm.zeros(BNE+1, bm.float64)                                                       
        eii[0] = bm.cumsum(ei)[-1] # 最后一个元素                                   
        eii[1:] = bm.cumsum(ei)                                                     
        value = eii[k]                                                              
                                                                                    
        # 计算第二部分的积分                                                        
        innode = bm.zeros(p.shape+(2,))                                             
        innode[..., 0, :] = bnode[k, 0]                                             
        innode[..., 1, :] = p                                                       
        emm = bm.sqrt((innode[...,0,0] - innode[...,1,0])**2 + (innode[...,0,1] - innode[...,1,1])**2)
        nnn = mesh.edge_unit_normal()[is_boundary_edge][index][k]                   
        bcpoint = bm.einsum('abcd,qc->abqd',innode, bcs)                            
        vvalue = bm.einsum('q, ..., ...d, ...qd->...', ws,emm,nnn, self.velocity_boundary(bcpoint))      
        value = value + vvalue                                                      
        return value      

    @cartesian
    def stream_function_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(x, bm.float64)


    @cartesian
    def pressure_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(p, bm.float64)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        flag1 = self.is_inlet_boundary(p)
        flag2 = self.is_wall_boundary(p)
        on_boundary = flag1 | flag2

        return on_boundary






def get_flist(u_sp, device=None): 
    x = sp.symbols("x")
    y = sp.symbols("y")

    #u_sp = sp.sin(4*x)*sp.cos(5*y)
    #u_sp = x*y
    ux_sp = sp.diff(u_sp, x)
    uy_sp = sp.diff(u_sp, y)
    uxx_sp = sp.diff(ux_sp, x)
    uyx_sp = sp.diff(ux_sp, y)
    uyy_sp = sp.diff(uy_sp, y)
    uxxx_sp = sp.diff(uxx_sp, x)
    uyxx_sp = sp.diff(uyx_sp, x)
    uyyx_sp = sp.diff(uyy_sp, x)
    uyyy_sp = sp.diff(uyy_sp, y)
    uxxxx_sp = sp.diff(uxxx_sp, x)
    print('uxxxx_sp:', uxxxx_sp)
    print(sp.simplify(uxxxx_sp))
    uyxxx_sp = sp.diff(uyxx_sp, x)
    print('uyxxx_sp:', uyxxx_sp)
    print(sp.simplify(uyxxx_sp))
    uyyxx_sp = sp.diff(uyyx_sp, x)
    print('uyyxx_sp:', uyyxx_sp)
    print(sp.simplify(uyyxx_sp))
    uyyyx_sp = sp.diff(uyyy_sp, x)
    print('uyyyx_sp:', uyyyx_sp)
    print(sp.simplify(uyyyx_sp))
    uyyyy_sp = sp.diff(uyyy_sp, y)
    print('uyyyy_sp:', uyyyy_sp)
    print(sp.simplify(uyyyy_sp))

    u     = sp.lambdify(('x', 'y'), u_sp, 'numpy') 
    ux    = sp.lambdify(('x', 'y'), ux_sp, 'numpy') 
    uy    = sp.lambdify(('x', 'y'), uy_sp, 'numpy') 
    uxx   = sp.lambdify(('x', 'y'), uxx_sp, 'numpy') 
    uyx   = sp.lambdify(('x', 'y'), uyx_sp, 'numpy') 
    uyy   = sp.lambdify(('x', 'y'), uyy_sp, 'numpy') 
    uxxx  = sp.lambdify(('x', 'y'), uxxx_sp, 'numpy') 
    uyxx  = sp.lambdify(('x', 'y'), uyxx_sp, 'numpy') 
    uyyx  = sp.lambdify(('x', 'y'), uyyx_sp, 'numpy') 
    uyyy  = sp.lambdify(('x', 'y'), uyyy_sp, 'numpy') 
    uxxxx = sp.lambdify(('x', 'y'), uxxxx_sp, 'numpy') 
    uyxxx = sp.lambdify(('x', 'y'), uyxxx_sp, 'numpy') 
    uyyxx = sp.lambdify(('x', 'y'), uyyxx_sp, 'numpy') 
    uyyyx = sp.lambdify(('x', 'y'), uyyyx_sp, 'numpy') 
    uyyyy = sp.lambdify(('x', 'y'), uyyyy_sp, 'numpy') 

    #f    = lambda node : u(node[..., 0], node[..., 1])
    def f(node):
        x = node[..., 0]
        y = node[..., 1]
        x_cpu = bm.to_numpy(x)
        y_cpu = bm.to_numpy(y)
        return bm.array(u(x_cpu, y_cpu), device=device)
    def grad_f(node):
        x = node[..., 0]
        y = node[..., 1]
        val = bm.zeros_like(node, device=device)
        x_cpu = bm.to_numpy(x) 
        y_cpu = bm.to_numpy(y)
        val[..., 0] = bm.array(ux(x_cpu, y_cpu), device=device)
        val[..., 1] = bm.array(uy(x_cpu, y_cpu), device=device)
        return val 
    def grad_2_f(node):
        x = node[..., 0]
        y = node[..., 1]
        val = bm.zeros(x.shape+(3, ), dtype=bm.float64, device=device)
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val[..., 0] = bm.array(uxx(x, y), device=device)
        val[..., 1] = bm.array(uyx(x, y), device=device)
        val[..., 2] = bm.array(uyy(x, y), device=device)
        return val 
    def grad_3_f(node):
        x = node[..., 0]
        y = node[..., 1]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(x.shape+(4, ), dtype=bm.float64, device=device)
        val[..., 0] = bm.array(uxxx(x, y), device=device)
        val[..., 1] = bm.array(uyxx(x, y), device=device)
        val[..., 2] = bm.array(uyyx(x, y), device=device)
        val[..., 3] = bm.array(uyyy(x, y), device=device)
        return val 
    def grad_4_f(node):
        x = node[..., 0]
        y = node[..., 1]
        x = bm.to_numpy(x)
        y = bm.to_numpy(y)
        val = bm.zeros(x.shape+(5, ), dtype=bm.float64, device=device)
        val[..., 0] = bm.array(uxxxx(x, y), device=device)
        val[..., 1] = bm.array(uyxxx(x, y), device=device)
        val[..., 2] = bm.array(uyyxx(x, y), device=device)
        val[..., 3] = bm.array(uyyyx(x, y), device=device)
        val[..., 4] = bm.array(uyyyy(x, y), device=device)
        return val 

    flist = [f, grad_f, grad_2_f, grad_3_f, grad_4_f]
    return flist
