import sympy as sp

from fealpy.backend import backend_manager as bm

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarMassIntegrator
from fealpy.mesh import TriangleMesh

class SymbolicIntegration:
    def __init__(self, space1, space2=None):
        self.space1 = space1
        self.mesh = space1.mesh
        self.p1 = space1.p  # 第一个空间的多项式次数
        self.GD = self.mesh.geo_dimension()  # 几何维度
        self.ldof1 = space1.number_of_local_dofs()  # 第一个空间的局部自由度数量
        self.mi1 = self.mesh.multi_index_matrix(p=self.p1, etype=self.GD)  # 第一个空间的多重指标矩阵

        # 如果没有提供第二个空间，则假设两个空间相同
        if space2 is None:
            self.space2 = space1
            self.p2 = self.p1
            self.ldof2 = self.ldof1
            self.mi2 = self.mi1
        else:
            self.space2 = space2
            self.p2 = space2.p  # 第二个空间的多项式次数
            self.ldof2 = space2.number_of_local_dofs()  # 第二个空间的局部自由度数量
            self.mi2 = self.mesh.multi_index_matrix(p=self.p2, etype=self.GD)  # 第二个空间的多重指标矩阵
        
        # 定义符号重心坐标 λ_i
        self.l = sp.symbols('l0:%d' % (self.GD+1), real=True)

    def basis(self, p, mi):
        GD = self.GD
        ldof = mi.shape[0]
        l = self.l
        
        # 初始化矩阵 A，大小为 (p+1) x (GD+1)
        A = sp.ones(p+1, GD+1)
        
        # 计算 A 矩阵的元素
        for i in range(1, p+1):
            for j in range(GD+1):
                A[i, j] = (p*l[j] - (i-1))*A[i-1, j]
            # 除以阶乘 i!
            A[i, :] /= sp.factorial(i)
        
        # 初始化基函数数组
        phi = sp.ones(ldof)
        
        # 计算基函数
        for i in range(ldof):
            phi_i = 1
            for j in range(GD+1):
                mi_ij = mi[i, j]
                phi_i *= A[mi_ij, j]
            phi[i] = phi_i
        
        return phi
    
    def multi_index(self, monomial):
        l = self.l
        GD = self.GD

        m = monomial.as_powers_dict()
        a = bm.zeros(GD+1, dtype=bm.int32) # 返回幂指标
        for i in range(GD+1):
            a[i] = int(m.get(l[i]) or 0)

        return a
    
    def integrate(self, f):
        GD = self.GD

        f = f.expand()
        r = 0    # 积分值
        for m in f.as_coeff_add()[1]:
            c = m.as_coeff_mul()[0] # 返回系数
            a = self.multi_index(m) # 返回单项式的幂指标
            temp = 1
            for i in range(GD+1):
                temp *= sp.factorial(a[i])
            r += sp.factorial(GD) * c * temp / sp.factorial(sum(a) + GD)

        return r + f.as_coeff_add()[0]
    
    def phi_phi_matrix(self):
        phi1 = self.basis(self.p1, self.mi1)
        phi2 = self.basis(self.p2, self.mi2)
        ldof1 = self.ldof1
        ldof2 = self.ldof2

        # 初始化矩阵 M
        M = sp.zeros(ldof1, ldof2)

        # 计算单元面积
        area = self.mesh.entity_measure('cell')[0]  # 获取第一个单元的面积

        for i in range(ldof1):
            for j in range(ldof2):
                integrand = phi1[i] * phi2[j]
                M[i, j] = self.integrate(integrand)
                integral_value = self.integrate(integrand)
                M[i, j] = integral_value * area  # 将面积乘到积分结果上

        return M

# 初始化网格和空间
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2, device='cpu')
# 定义两个不同的多项式次数
p1 = 1
p2 = 1

# 创建两个有限元空间
space1 = LagrangeFESpace(mesh=mesh, p=p1, ctype='C')
space2 = LagrangeFESpace(mesh=mesh, p=p2, ctype='C')

# 创建符号积分类的实例，传入两个空间
symbolic_int = SymbolicIntegration(space1, space2)

# 计算质量矩阵
M = symbolic_int.phi_phi_matrix()

integrator = ScalarMassIntegrator(q=5)
M1 = integrator.assembly(space=space1)
print("M1:", M1[0])

# 输出质量矩阵
print("Mass Matrix M:")
sp.pprint(M)