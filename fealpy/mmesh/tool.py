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
        
class Poissondata():
    def __init__(self , u :str ,var_list: list[str], D = [0,1,0,1]):
        u = sympify(u)
        self.TD = len(var_list)
        x = symbols(var_list[0])
        y = symbols(var_list[1])
        z = symbols(var_list[-1])
        self.u = lambdify(var_list, u ,'numpy')
        f_str = -diff(u,x,2) - diff(u,y,2)
        if self.TD == 3:
            f_str -= diff(u,z,2)
        self.f = lambdify(var_list,  f_str)
        self.grad_ux = lambdify(var_list, diff(u,x,1))
        self.grad_uy = lambdify(var_list, diff(u,y,1))
        self.grad_uz = lambdify(var_list, diff(u,z,1))
        self.domain = D

    def domain(self):
        return self.domain
    
    def solution(self, p):
        x = p[...,0]
        y = p[...,1]
        if self.TD == 3:
            z = p[...,2]
            return self.u(x,y,z)
        return self.u(x,y)
    
    def init_solution(self, p):
        return self.solution(p)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        if self.TD == 3:
            z = p[...,2]
            return self.f(x,y,z)
        return self.f(x,y)
    
    def gradient(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros_like(p)
        if self.TD == 3:
            z = p[...,2]
            val[...,0] = self.grad_ux(x,y,z)
            val[...,1] = self.grad_uy(x,y,z)
            val[...,2] = self.grad_uz(x,y,z)
        val[...,0] = self.grad_ux(x,y)
        val[...,1] = self.grad_uy(x,y)
        return val
    
    def dirichlet(self,p ):
        return self.solution(p)
    
# visualize the mesh and solution   

def high_order_meshploter(ax,mesh, uh= None, model='mesh', scat_node=True , scat_index = slice(None)):
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
    edges = mesh.edge
    if model == 'mesh':
        p = nodes[edges]
        curve = lagrange_interpolation(p, n=n)
        ax.plot(curve[..., 0], curve[..., 1], 'b-', linewidth=0.5)  # 使用 ax 绘制曲线
        if scat_node:
            ax.scatter(nodes[scat_index, 0], 
                       nodes[scat_index, 1], s=1, color='r')  # 使用 ax 绘制节点
        ax.set_aspect('equal')  # 设置坐标轴比例
        ax.axis('off')  # 关闭坐标轴

    elif model == 'surface':
        points = bm.concat([nodes, uh[...,None]], axis=-1)
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

def linear_surfploter(ax,mesh, uh,scat_node =False , scat_index = slice(None),alpha =0.7):
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    if isinstance(mesh, TriangleMesh):
        ax.plot_trisurf(node[:, 0], node[:, 1], uh[:], 
                        triangles=cell, cmap='viridis', edgecolor='blue',linewidth=0.15,alpha = alpha)
        
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
        quad_mesh = Poly3DCollection(verts, alpha=alpha, edgecolor='b', linewidth=0.15)
        quad_mesh.set_facecolor(colors)
        ax.add_collection3d(quad_mesh)
    if scat_node:
        ax.scatter(node[scat_index, 0], node[scat_index, 1], uh[scat_index],s = 2, color='r')  # 绘制节点
    # ax.clear()


import matplotlib.pyplot as plt
class AnimationTool: 
    def __init__(self, update_func, frames, save_path=None, fps=30, fig=None, **kwargs):
        """
        @brief 动图工具类
        @param update_func: 更新函数，接受一个参数 frame
        @param frames: 动画的帧数或帧生成器
        @param save_path: 保存路径，如果为 None 则不保存
        @param fps: 动画帧率
        @param fig: 可选的 matplotlib Figure 对象，如果为 None 则自动创建
        @param kwargs: 传递给 plt.figure 的其他参数
        """
        
        self.update_func = update_func
        self.frames = frames
        self.save_path = save_path
        self.fps = fps
        self.fig = fig or plt.figure(**kwargs)
        # self.ax = self.fig.add_subplot(111, projection='3d') if 'projection' in kwargs and \
        #                             kwargs['projection'] == '3d' else self.fig.add_subplot()

    def run(self):
        from matplotlib.animation import FuncAnimation, PillowWriter
        """
        @brief 运行动画
        """
        ani = FuncAnimation(self.fig, self.update_func, frames=self.frames, repeat=False)
        if self.save_path:
            if self.save_path.endswith('.gif'):
                # 使用 PillowWriter 保存为 GIF
                ani.save(self.save_path, writer=PillowWriter(fps=self.fps))
            else:
                # 默认保存为 MP4
                ani.save(self.save_path, writer='ffmpeg', fps=self.fps)
        else:
            plt.show()

def segmenter(array,dim,index):
    """
    @brief partition the array into dim segments
    @param array: the input array
    @param dim: the number of segments
    @param index: the index of the segment to be extracted
    @return: the extracted segment
    """
    n = array.shape[0]
    segment_size, remainder = divmod(n, dim)
    start = index * segment_size + min(index, remainder)
    end = start + segment_size + (1 if index < remainder else 0)
    return array[start:end]

def quad_equ_solver(coef:list):
    """
    @brief solve the quadratic equation
    @param a: the coefficient of x^2
    @param b: the coefficient of x
    @param c: the constant term
    """
    eps = 1e-12
    a,b,c = coef
    is_linear = bm.abs(a) < eps
    if bm.any(is_linear):
        linear_mask = is_linear
        x_linear = -c[linear_mask] / (b[linear_mask] + eps)
    x1 = None
    x2 = None
    quadratic_mask = ~is_linear
    if bm.any(quadratic_mask):
        a_mark = a[quadratic_mask]
        b_mark = b[quadratic_mask]
        c_mark = c[quadratic_mask]
        discriminant = b_mark**2 - 4*a_mark*c_mark
        right_idx = bm.where(discriminant >= 0)[0]
        x1 = (-b_mark[right_idx] + bm.sqrt(discriminant[right_idx]))/(2*a_mark[right_idx])
        x2 = (-b_mark[right_idx] - bm.sqrt(discriminant[right_idx]))/(2*a_mark[right_idx])
    if x1 is None:
        x=  x_linear
    else:
        x = bm.concat([x1,x2])
        if bm.any(is_linear):
            x = bm.concat([x, x_linear])
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

def _compute_coef_general_2d(A,C):
    """
    @brief compute the coefficient of the quadratic equation
    """
    a = bm.linalg.det(C)
    c = bm.linalg.det(A)
    b = (A[:, 0, 0] * C[:, 1, 1] - 
            A[:, 0, 1] * C[:, 1, 0] + 
            C[:, 0, 0] * A[:, 1, 1] - 
            C[:, 0, 1] * A[:, 1, 0])
    return [a, b, c]

def _compute_coef_general_3d(A,C):
    """
    @brief compute the coefficient of the cubic equation
    """
    a0, a1, a2 = (C[:, 1, 1] * C[:, 2, 2] - C[:, 1, 2] * C[:, 2, 1],
                    C[:, 1, 2] * C[:, 2, 0] - C[:, 1, 0] * C[:, 2, 2],
                    C[:, 1, 0] * C[:, 2, 1] - C[:, 1, 1] * C[:, 2, 0])
    b0, b1, b2 = (A[:, 1, 1] * C[:, 2, 2] - A[:, 1, 2] * C[:, 2, 1] + 
                    C[:, 1, 1] * A[:, 2, 2] - C[:, 1, 2] * A[:, 2, 1], 
                    A[:, 1, 0] * C[:, 2, 2] - A[:, 1, 2] * C[:, 2, 0] + 
                    C[:, 1, 0] * A[:, 2, 2] - C[:, 1, 2] * A[:, 2, 0], 
                    A[:, 1, 0] * C[:, 2, 1] - A[:, 1, 1] * C[:, 2, 0] + 
                    C[:, 1, 0] * A[:, 2, 1] - C[:, 1, 1] * A[:, 2, 0])
    c0, c1, c2 = (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1], 
                    A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0],
                    A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0])
    a = C[:, 0, 0] * a0 - C[:, 0, 1] * a1 + C[:, 0, 2] * a2
    ridx = bm.where(a > 1e-14)[0]
    b = (A[:, 0, 0] * a0 - A[:, 0, 1] * a1 + 
            A[:, 0, 2] * a2 + C[:, 0, 0] * b0 - 
            C[:, 0, 1] * b1 + C[:, 0, 2] * b2)
    c = (A[:, 0, 0] * b0 - A[:, 0, 1] * b1 + 
            A[:, 0, 2] * b2 + C[:, 0, 0] * c0 - 
            C[:, 0, 1] * c1 + C[:, 0, 2] * c2)
    d = A[:, 0, 0] * c0 - A[:, 0, 1] *c1 + A[:, 0, 2] * c2
    a, b, c, d = a[ridx], b[ridx], c[ridx], d[ridx]
    return [a, b, c, d]

def _compute_coef_2d(delta_x,AC_gererator):
    """
    Compute coefficients for 2D case.
    """
    return _compute_general_coef(delta_x,AC_gererator,_compute_coef_general_2d)

def _compute_coef_3d(delta_x,AC_gererator):
    """
    Compute coefficients for 3D case.
    """
    return _compute_general_coef(delta_x,AC_gererator, _compute_coef_general_3d)

def _compute_general_coef(delta_x ,AC_gererator, fun):
    """
    @brief compute the coefficient of the quadratic equation
    """
    A, C = AC_gererator(delta_x)
    return fun(A, C)

def _solve_bilinear_system(p, A, B, C, D):
    """
    @brief 统一的双线性系统求解 [0,1]×[0,1] 参数空间
    """
    px, py = p[:, 0], p[:, 1]
    Ax, Ay = A[:, 0], A[:, 1]
    Bx, By = B[:, 0], B[:, 1]
    Cx, Cy = C[:, 0], C[:, 1]
    Dx, Dy = D[:, 0], D[:, 1]
    
    kwarg = bm.context(px)
    eps = 1e-12
    
    # 选择更稳定的方程（基于Cx, Cy的绝对值）
    use_x_eq = bm.abs(Cx) >= bm.abs(Cy)
    
    # 统一的参数选择，避免重复的if-else逻辑
    p_coord = bm.where(use_x_eq, px, py)
    p_other = bm.where(use_x_eq, py, px)
    A_coord = bm.where(use_x_eq, Ax, Ay)
    A_other = bm.where(use_x_eq, Ay, Ax)
    B_coord = bm.where(use_x_eq, Bx, By)
    B_other = bm.where(use_x_eq, By, Bx)
    C_coord = bm.where(use_x_eq, Cx, Cy)
    C_other = bm.where(use_x_eq, Cy, Cx)
    D_coord = bm.where(use_x_eq, Dx, Dy)
    D_other = bm.where(use_x_eq, Dy, Dx)
    
    # 计算二次方程系数（统一公式）
    delta_other = A_other - p_other
    delta_coord = A_coord - p_coord 
    
    a = B_other * D_coord - D_other * B_coord
    b = B_other * C_coord - C_other * B_coord + D_coord * delta_other - D_other * delta_coord
    c = delta_other * C_coord - delta_coord * C_other
    
    # 求解二次方程（保留eta验证）
    xi = _solve_quadratic_with_validation(
        a, b, c, p_coord, A_coord, B_coord, C_coord, D_coord, eps
    )
    
    # 计算eta
    eta = (p_coord - A_coord - B_coord * xi) / (C_coord + D_coord * xi + eps)
    
    # 组装结果
    xi_eta = bm.zeros((p.shape[0], 2), **kwarg)
    xi_eta = bm.set_at(xi_eta, (..., 0), xi)
    xi_eta = bm.set_at(xi_eta, (..., 1), eta)
    
    return xi_eta

def _solve_quadratic_with_validation(a, b, c, p_coord, A_coord, B_coord, C_coord, D_coord, eps):
    """
    @brief 改进的二次方程求解，同时验证对应的eta是否合理
    """
    # 处理线性情况
    is_linear = bm.abs(a) < eps
    xi = bm.zeros_like(a)
    
    # 线性情况
    if bm.any(is_linear):
        linear_mask = is_linear
        xi_linear = -c[linear_mask] / (b[linear_mask] + eps)
        
        # 验证对应的eta
        eta_linear = ((p_coord[linear_mask] - A_coord[linear_mask] - B_coord[linear_mask] * xi_linear) / 
                     (C_coord[linear_mask] + D_coord[linear_mask] * xi_linear + eps))
        
        # 检查(xi, eta)对是否都在合理范围内
        tolerance = 1e-6  # 放宽容差
        pair_valid = ((xi_linear >= -tolerance) & (xi_linear <= 1.0 + tolerance) &
                     (eta_linear >= -tolerance) & (eta_linear <= 1.0 + tolerance))
        print(f"Linear valid pairs: {bm.sum(pair_valid)} out of {bm.size(pair_valid)}")
        # 无效的解设为中心点
        xi_linear = bm.where(pair_valid, xi_linear, 0.5)
        xi = bm.set_at(xi, linear_mask, xi_linear)

    # 二次情况
    quadratic_mask = ~is_linear
    if bm.any(quadratic_mask):
        mask = quadratic_mask
        discriminant = b[mask]**2 - 4*a[mask]*c[mask]
        
        # 只处理有实解的情况
        has_real_roots = discriminant >= 0
        if bm.any(has_real_roots):
            real_mask = mask.copy()
            real_mask[mask] = has_real_roots
            
            sqrt_d = bm.sqrt(discriminant[has_real_roots])
            a_real = a[real_mask]
            b_real = b[real_mask]
            
            x1 = (-b_real + sqrt_d) / (2 * a_real)
            x2 = (-b_real - sqrt_d) / (2 * a_real)
            
            # 计算对应的eta值
            eta1 = ((p_coord[real_mask] - A_coord[real_mask] - B_coord[real_mask] * x1) / 
                   (C_coord[real_mask] + D_coord[real_mask] * x1 + eps))
            eta2 = ((p_coord[real_mask] - A_coord[real_mask] - B_coord[real_mask] * x2) / 
                   (C_coord[real_mask] + D_coord[real_mask] * x2 + eps))
            
            # 检查(xi, eta)对的有效性
            tolerance = 1e-6
            x1_pair_valid = ((x1 >= -tolerance) & (x1 <= 1.0 + tolerance) &
                            (eta1 >= -tolerance) & (eta1 <= 1.0 + tolerance))
            x2_pair_valid = ((x2 >= -tolerance) & (x2 <= 1.0 + tolerance) &
                            (eta2 >= -tolerance) & (eta2 <= 1.0 + tolerance))
            
            # 选择策略：优先选择(xi, eta)都在范围内的解
            xi_quad = bm.where(x1_pair_valid & ~x2_pair_valid, x1,           # 只有x1有效
                              bm.where(x2_pair_valid & ~x1_pair_valid, x2,   # 只有x2有效
                              bm.where(x1_pair_valid & x2_pair_valid,        # 都有效时
                              bm.where(bm.abs(x1 - 0.5) < bm.abs(x2 - 0.5), x1, x2),
                              bm.where(bm.abs(x1 - 0.5) < bm.abs(x2 - 0.5), x1, x2))))# 中心选择
            
            xi = bm.set_at(xi, real_mask, xi_quad)
    
    return xi

def _solve_quad_parametric_coords(target_points, quad_vertices):
    """
    直接代数求解四边形参数坐标 (ξ,η) ∈ [0,1]x[0,1]

    Parameter:
        target_points: 目标点坐标，形状为 (N, 2)
        quad_vertices: 四边形顶点坐标，形状为 (N, 4, 2)，四个顶点按逆时针顺序排列
    """
    # 四边形顶点按逆时针顺序：v0(0,0), v1(1,0), v2(1,1), v3(0,1)
    v0, v1, v2, v3 = quad_vertices[:, 0], quad_vertices[:, 1], quad_vertices[:, 2], quad_vertices[:, 3]
    p = target_points
    
    # 双线性映射：x(ξ,η) = A + Bξ + Cη + Dξη
    A = v0                          # 常数项
    B = v1 - v0                     # ξ 项系数
    C = v3 - v0                     # η 项系数  
    D = v2 - v1 - v3 + v0          # ξη 项系数
    
    # 求解方程组：p = A + Bξ + Cη + Dξη
    D_norm = bm.linalg.norm(D, axis=-1)
    is_linear = D_norm < 1e-10
    
    xi_eta = bm.zeros((target_points.shape[0], 2), **bm.context(target_points))
    
    # 处理线性情况（平行四边形）
    if bm.any(is_linear):
        linear_mask = is_linear
        rhs = p[linear_mask] - A[linear_mask]
        matrix = bm.stack([B[linear_mask], C[linear_mask]], axis=-1)
        try:
            solution = bm.linalg.solve(matrix, rhs[..., None])[..., 0]
            xi_eta = bm.set_at(xi_eta, linear_mask, solution)
        except:
            pass
    # 处理非线性情况
    if bm.any(~is_linear):
        xi_eta_nl = _solve_bilinear_system(
            p[~is_linear], A[~is_linear], B[~is_linear], 
            C[~is_linear], D[~is_linear]
        )
        xi_eta = bm.set_at(xi_eta, ~is_linear, xi_eta_nl)

    return xi_eta
