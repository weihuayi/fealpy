from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d
from typing import Sequence


class Exp0006(BoxMesher2d):
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

    def init_mesh(self):
        #box = [0, 1, 0, 1]
        box = [0, 2.2, 0, 0.41]
        center = [0.205, 0.205]
        #center = [0.2, 0.1]
        radius = 0.05
        #h = 0.5
        h = 0.004
        #n_circle = 168
        n_circle = 1000
        #n_circle = 32
        #n_circle = 4
        self.box = box
        self.center = center
        self.radius = radius

        from meshpy.triangle import MeshInfo, build
        from fealpy.mesh import TriangleMesh
        points = [
            (box[0], box[2]),
            (box[1], box[2]),
            (box[1], box[3]),
            (box[0], box[3])
        ]

        # 矩形边界
        facets = [[0, 1], [1, 2], [2, 3], [3, 0]]

        # 圆的离散点
        cx, cy = center
        theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False)
        circle_points = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]

        # 圆的边界：顺时针编号（meshpy 要求空洞边界为顺时针）
        circle_facets = [[i, (i+1) % n_circle] for i in range(n_circle)]

        # 合并点和边界
        circle_offset = len(points)
        all_points = points + circle_points
        all_facets = facets + [[i[0]+circle_offset, i[1]+circle_offset] for i in circle_facets]

        # 设置空洞区域（空洞内一点）
        hole_point = [cx, cy]

        # meshpy 生成三角网格
        mesh_info = MeshInfo()
        mesh_info.set_points(all_points)
        mesh_info.set_facets(all_facets)
        mesh_info.set_holes([hole_point])  # 空洞位置

        mesh = build(mesh_info, max_volume=h**2)

        node = bm.array(mesh.points)
        cell = bm.array(mesh.elements)

        # 转为 FEALPy 的 TriangleMesh
        return TriangleMesh(node, cell)

    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:                   
        """Check if point where velocity is defined is on boundary."""          
        x = p[..., 0]                                                           
        y = p[..., 1]                                                           
        atol = 1e-12                                                            
        # 检查是否接近 x=±1 或 y=±1                                             
        on_boundary = (                                                         
            (bm.abs(x - self.box[0]) < atol) &                                  
            (y > self.box[2]) & (y < self.box[3]))                              
        return on_boundary 

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (bm.abs(x - self.box[1]) < atol)
        return on_boundary

    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(y - self.box[2]) < atol) | (bm.abs(y - self.box[3]) < atol))
        return on_boundary

    @cartesian                                                                  
    def is_cylinder_boundary(self, p: TensorLike) -> TensorLike:                
        """Check if point where velocity is defined is on boundary."""          
        x = p[..., 0]                                                           
        y = p[..., 1]                                                           
        cx, cy = self.center                                                    
        radius = self.radius                                                    
        atol = 1e-4                                                        
        # 检查是否接近圆的边界                                                  
        on_boundary = bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol      
        return on_boundary 

    @cartesian
    def velocity_boundary(self, p):
        x, y = p[..., 0], p[..., 1]
        eps = 1e-12 
        flag1 = self.is_inlet_boundary(p)
        flag2 = self.is_outlet_boundary(p)
        flag3 = self.is_wall_boundary(p)
        #flag4 = self.is_cylinder_boundary(p)
        u = bm.zeros_like(p)
        u[flag1, 0] = 1.2 * y[flag1] * (0.41 - y[flag1])/(0.41**2)
        u[flag2, 0] = 1.2 * y[flag2] * (0.41 - y[flag2])/(0.41**2)
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
        is_boundary_edge = self.is_wall_boundary(menode) | self.is_inlet_boundary(menode) | self.is_outlet_boundary(menode) | self.is_cylinder_boundary(menode)  # 只有外边界
        #is_boundary_edge = mesh.boundary_edge_flag()                                
        #is_boundary_edge = is_boundary_edge & isnedge
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
    def stream_function_boundary1(self, p):                                                       
                                                                                    
        ## 判断点在第几个边界上                                                     
        value = bm.zeros_like(p.shape[:-1])                                         
        mesh = self.init_mesh() 
        node = mesh.entity('node')
        edge = mesh.entity('edge')                                                  
        enode = node[edge]
        menode = (enode[:, 0, :] + enode[:, 1, :])/2

        #is_boundary_edge = mesh.boundary_edge_flag()                                
        is_boundary_edge = self.is_wall_boundary(menode) | self.is_inlet_boundary(menode) | self.is_outlet_boundary(menode) | self.is_cylinder_boundary(menode)  # 只有外边界
        bedge, index = self.order_edge(0)                                          
        bnode = mesh.entity('node')[bedge]                                          
        BNE = bedge.shape[0]                                                        
        k = bm.zeros(p.shape[:-1], dtype=bm.int32)                                  
        eps = 1e-10                                                                 
        for i in range(BNE):                                                        
            PA = p - bnode[i, 0, :]                                                 
            PB = p - bnode[i, 1, :]                                                 
            flag1 = bm.abs(bm.cross(PA, PB)) <= eps                                 
            flag2 = bm.sum(PA*PB,axis=-1 ) <= 0                                     
            flag = flag1 & flag2                                                    
            k[flag] = i                                                             
        # 计算边积分                                                                
        q = 9                                                                       
        qf = mesh.quadrature_formula(q, 'edge')                                     
        bcs, ws = qf.get_quadrature_points_and_weights()                            
        point = bm.bc_to_points(bcs, node, entity = bedge) #(BNE, NQ, GD)           
        ei = bm.zeros_like(bedge)                                                   
        en = mesh.edge_unit_normal()[is_boundary_edge][index]                       
        em = mesh.edge_length()[is_boundary_edge][index]                            
        ei = bm.einsum('eqd,q,e,ed->e', self.velocity_boundary(point),ws,em,en)                          
                                                                                    
        # 计算第一部分的积分                                                        
        eii = bm.zeros(BNE+1)                                                       
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
    def stream_function_boundary(self, p):
        #p = bm.array([0.25374, 0.21612])
        is_boundary_edge = self.is_cylinder_boundary(p)
        value = self.stream_function_boundary1(bm.array([[[0, 0.1]]]))
        value1 = self.stream_function_boundary1(p)
        value1[is_boundary_edge] = bm.float64(value)
        return value1


    @cartesian
    def stream_function_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(x)


    @cartesian
    def pressure_source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        flag1 = self.is_inlet_boundary(p)
        flag2 = self.is_outlet_boundary(p)
        flag3 = self.is_wall_boundary(p)
        flag4 = self.is_cylinder_boundary(p)
        on_boundary = flag1 | flag2 | flag3 | flag4

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
