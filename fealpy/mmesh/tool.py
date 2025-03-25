from .config import *
from sympy import *

class MeshGenerator():
    def __init__(self, nx, ny, nz = None ,box=[0,1,0,1], meshtype='tri', p = 1,delete_bound=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.box = box
        self.meshtype = meshtype
        self.p = p
        self.delete_bound = delete_bound

    def __call__(self, *args, **kwds):
        mesh = self.get_mesh()
        return mesh
    
    def thr(self,p):
        x = p[...,0]
        y=  p[...,1]
        area = self.delete_bound
        in_x = (x >= area[0]) & (x <= area[1])
        in_y = (y >= area[2]) & (y <= area[3])
        if p.shape[-1] == 3:
            z = p[...,2]
            in_z = (z >= area[4]) & (z <= area[5])
            return in_x & in_y & in_z
        return  in_x & in_y
    
    @classmethod
    def get_mesh(cls ,  nx, ny, nz = None ,box=[0,1,0,1], meshtype='tri',p = 1,delete_bound=None):
        instance = cls(nx, ny, nz, box, meshtype,p, delete_bound)
        if instance.delete_bound is  None:
            thr = None
        else:
            thr = instance.thr
        if instance.meshtype == 'tri':
            return TriangleMesh.from_box( instance.box,instance.nx, instance.ny,threshold=thr)
        elif instance.meshtype == 'quad':
            return QuadrangleMesh.from_box(instance.box, instance.nx, instance.ny, threshold=thr)
        elif instance.meshtype == 'tet':
            if instance.nz is None:
                raise ValueError("nz must be given")
            if len(instance.box) != 6:
                raise ValueError("box is not correct")
            return TetrahedronMesh.from_box(instance.box, instance.nx, instance.ny, instance.nz, threshold=thr)
        elif instance.meshtype == 'hex':
            if instance.nz is None:
                raise ValueError("nz must be given")
            if len(instance.box) != 6:
                raise ValueError("box is not correct")
            return HexahedronMesh.from_box(instance.box,instance.nx, instance.ny, instance.nz, threshold=thr)
        elif instance.meshtype == 'lagtri':
            tri = TriangleMesh.from_box(instance.box, instance.nx, instance.ny , threshold=thr)
            return LagrangeTriangleMesh.from_triangle_mesh(tri, p=instance.p)
        elif instance.meshtype == 'lagquad':
            quad = QuadrangleMesh.from_box(instance.box, instance.nx, instance.ny , threshold=thr)
            return LagrangeQuadrangleMesh.from_quadrangle_mesh(quad, p=instance.p)
        else:
            raise ValueError("meshtype must be tri, quad, tet or hex")

# visualize the mesh and solution   

def high_order_meshploter(mesh, uh= None, model='mesh', scat_node=True , scat_index = slice(None)):
    import matplotlib.pyplot as plt
    nodes = mesh.node
    n = mesh.p
    def lagrange_interpolation(points, num_points=100, n = n):
        """
        @brief 利用拉格朗日插值构造曲线
        @param points: 插值点的列表 [(x0, y0), (x1, y1), ..., (xp, yp)]
        @param num_points: 曲线上点的数量
        @param n: 插值多项式的次数
        @return: 曲线上点的坐标数组
        """
        t = bm.linspace(0, n, num_points)

        def L(k, t):
            Lk = bm.ones_like(t)
            for i in range(n + 1):
                if i != k:
                    Lk *= (t - i) / (k - i)
            return Lk
        GD = points.shape[-1]
        curve = bm.zeros((t.shape[0], points.shape[0],GD), dtype=bm.float64)
        for k in range(n + 1):
            Lk = L(k, t)
            for i in range(GD):
                xk = points[:,k,i]
                bm.add_at(curve , (...,i) ,Lk[:,None] * xk)
        return curve
    fig = plt.figure()
    edges = mesh.edge
    if model == 'mesh':
        p = nodes[edges]
        curve = lagrange_interpolation(p,n=n)
        plt.plot(curve[...,0], curve[...,1], 'b-',linewidth=0.5)
        if scat_node:
            plt.scatter(nodes[scat_index, 0], 
                        nodes[scat_index, 1],s = 3, color='r')  # 绘制节点
        plt.gca().set_aspect('equal')
        plt.axis('off') # 关闭坐标轴
    elif model == 'surface':
        points = bm.concat([nodes, uh[...,None]], axis=-1)
        ax = fig.add_subplot(111, projection='3d')
        p = points[edges]
        curve = lagrange_interpolation(p,n=n).transpose(1,0,2)
        nan_separator = bm.full((curve.shape[0], 1, 3), bm.nan)  # 每条曲线之间插入 NaN
        concatenated = bm.concat([curve, nan_separator], axis=1)  # 插入分隔符
        curve = concatenated.reshape(-1, 3) 
        ax.plot(curve[..., 0], curve[..., 1], curve[..., 2], linewidth=1)
        if scat_node:
            ax.scatter(points[scat_index, 0], 
                       points[scat_index, 1], 
                       points[scat_index, 2],s = 2, color='r')  # 绘制节点
    plt.show()

def linear_surfploter(mesh, uh,scat_node =False , scat_index = slice(None)):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    if isinstance(mesh, TriangleMesh):
        ax.plot_trisurf(node[:, 0], node[:, 1], uh[:], triangles=cell, cmap='viridis', edgecolor='blue',linewidth=0.15)
        
    elif isinstance(mesh, QuadrangleMesh):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        xyz = bm.concat((node, uh[:,None]), axis=-1)
        verts = xyz[cell]
        
        uh_mean = bm.mean(uh[cell], axis=1)
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=uh_mean.min(), vmax=uh_mean.max())
        colors = cmap(norm(uh_mean))
        quad_mesh = Poly3DCollection(verts, alpha=0.9, edgecolor='b', linewidth=0.15)
        quad_mesh.set_facecolor(colors)
        ax.add_collection3d(quad_mesh)
    if scat_node:
        ax.scatter(node[scat_index, 0], node[scat_index, 1], uh[scat_index],s = 2, color='r')  # 绘制节点
    plt.show()

def quad_equ_solver(coef:list):
    """
    @brief solve the quadratic equation
    @param a: the coefficient of x^2
    @param b: the coefficient of x
    @param c: the constant term
    """
    a,b,c = coef
    discriminant = b**2 - 4*a*c
    right_idx = bm.where(discriminant >= 0)[0]
    x1 = (-b[right_idx] + bm.sqrt(discriminant[right_idx]))/(2*a[right_idx])
    x2 = (-b[right_idx] - bm.sqrt(discriminant[right_idx]))/(2*a[right_idx])
    x = bm.concat([x1,x2])
    return x
    
def cubic_equ_solver(coef:list):
    """
    @brief solve the cubic equation
    @param a: the coefficient of x^3
    @param b: the coefficient of x^2
    @param c: the coefficient of x
    @param d: the constant term
    """
    a,b,c,d = coef
    kwarg = bm.context(a)
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
    discriminant = (q / 2)**2 + (p / 3)**3
    u1 = (-q / 2 + bm.sqrt(discriminant))**(1 / 3)
    u2 = (-q / 2 - bm.sqrt(discriminant))**(1 / 3)
    v1 = (-q / 2 - bm.sqrt(discriminant))**(1 / 3)
    v2 = (-q / 2 + bm.sqrt(discriminant))**(1 / 3)
    u = bm.where(discriminant >= 0, u1,u2)
    v = bm.where(discriminant >= 0, v1,v2)
    upv = u + v
    umv = u - v
    move_dis = b / (3 * a)
    x1 = upv - move_dis
    sq3 = bm.sqrt(bm.array([3]),**kwarg)
    x2 = -upv / 2 + umv * sq3 * 1j / 2 - move_dis
    x3 = -upv / 2 - umv * sq3 * 1j / 2 - move_dis
    x = bm.concat([x1, x2, x3])
    return x