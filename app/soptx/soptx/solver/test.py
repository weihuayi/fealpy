import sympy as sp

from fealpy.backend import backend_manager as bm

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator
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
        """计算p次拉格朗日基函数
        
        使用递推方法构造基函数，可以得到正确的系数和形式。
        例如，对于 p=3 时：
        - 顶点基函数：λᵢ(3λᵢ - 2)(3λᵢ - 1)/4
        - 边内部点基函数：9λᵢλⱼ(3λᵢ - 1)/2
        - 内部点基函数：27λ₀λ₁λ₂
        
        Parameters
        ----------
        p : int
            多项式次数
        mi : ndarray
            多重指标矩阵，形状为 (ldof, GD+1)
            
        Returns
        -------
        phi : list
            基函数列表
        """
        GD = self.GD
        ldof = mi.shape[0]
        l = self.l
        
        # 初始化矩阵 A，大小为 (p+1) x (GD+1)
        A = sp.ones(p+1, GD+1)
        
        # 递推计算 A 矩阵的元素
        for i in range(1, p + 1):
            for j in range(GD + 1):
                # 构造形如 (pλ - (i-1)) 的项
                A[i, j] = (p*l[j] - (i-1))*A[i-1, j]
        for i in range(1, p+1):
            # 除以阶乘 i! 来得到正确的系数
            A[i, :] /= sp.factorial(i)
        
        # 构造基函数
        phi = []
        for i in range(ldof):
            phi_i = 1
            for j in range(GD + 1):
                mi_ij = mi[i, j]
                phi_i *= A[mi_ij, j]
            phi.append(phi_i)
        
        return phi

    def basis_old(self, p, mi):
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
    
    def grad_basis(self, p, mi):
        phi = self.basis(p, mi)  # 获取基函数
        ldof = len(phi)
        grad_phi = []
        
        # 对每个基函数计算导数
        for i in range(ldof):
            # 计算对每个重心坐标的导数
            grad_i = [sp.diff(phi[i], self.l[j]) for j in range(self.GD+1)]
            grad_phi.append(grad_i)
            
        return grad_phi
    
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

        M = sp.tensor.array.MutableDenseNDimArray(
            sp.zeros(ldof1 * ldof2),
            (1, ldof1, ldof2)
            )

        for i in range(ldof1):
            for j in range(ldof2):
                integrand = phi1[i] * phi2[j]
                M[0, i, j] = self.integrate(integrand)

        return M
    

    def gphi_gphi_matrix(self):
        # 计算两个空间基函数的导数
        l = self.l
        phi1 = self.basis(self.p1, self.mi1)
        phi2 = self.basis(self.p2, self.mi2)
        gphi1 = self.grad_basis(self.p1, self.mi1)
        gphi2 = self.grad_basis(self.p2, self.mi2)
        
        # 初始化结果矩阵
        S = sp.tensor.array.MutableDenseNDimArray(
            sp.zeros(self.ldof1 * self.ldof2 * (self.GD+1) * (self.GD+1)),
            (self.ldof1, self.ldof2, self.GD+1, self.GD+1)
            )
        
        # 计算所有方向导数的组合
        for i in range(self.ldof1):
            for j in range(self.ldof2):
                for m in range(self.GD + 1):
                    for n in range(self.GD + 1):
                        # gphi1 = sp.diff(phi1[i],l[m])
                        # gphi2 = sp.diff(phi2[j],l[n])
                        # temp= sp.diff(phi1[i],l[m]) * sp.diff(phi2[j],l[n])
                        temp = gphi1[i][m] * gphi2[j][n]
                        S[i, j, m, n] = self.integrate(temp)
                    
        return S

# 初始化网格和空间
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=2, ny=2, device='cpu')
# 定义两个不同的多项式次数
p1, p2 = 3, 3

# 创建两个有限元空间
space1 = LagrangeFESpace(mesh=mesh, p=p1, ctype='C')
space2 = LagrangeFESpace(mesh=mesh, p=p2, ctype='C')

# 创建符号积分类的实例，传入两个空间
symbolic_int = SymbolicIntegration(space1, space2)

q = p1+1
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
gphi_lambda = space1.grad_basis(bcs,variable='u')
M = bm.einsum('q, qik, qjl->ijkl', ws, gphi_lambda, gphi_lambda)

sphiu = symbolic_int.basis(p1, symbolic_int.mi1)
print("sphiu:", sphiu)
sgphiu = symbolic_int.grad_basis(p1, symbolic_int.mi1)
print("sgphiu:", sgphiu)
gphiu_gphiu_11 = symbolic_int.gphi_gphi_matrix()
print("gphiu_gphiu_11:\n", gphiu_gphiu_11)

error = bm.sum(bm.abs(M - gphiu_gphiu_11))
print(f"error: {error}")

glambda_x = mesh.grad_lambda()
# print("glambda_x:\n", glambda_x)

cm = mesh.entity_measure('cell')
gphix_gphix_11 = bm.einsum('ijkl, ck, cl, c -> cij', gphiu_gphiu_11, glambda_x[..., 0], glambda_x[..., 0], cm)
gphiy_gphiy_11 = bm.einsum('ijkl, ck, cl, c -> cij', gphiu_gphiu_11, glambda_x[..., 1], glambda_x[..., 1], cm)
gphix_gphiy_11 = bm.einsum('ijkl, ck, cl, c -> cij', gphiu_gphiu_11, glambda_x[..., 0], glambda_x[..., 1], cm)
gphiy_gphix_11 = bm.einsum('ijkl, ck, cl, c -> cij', gphiu_gphiu_11, glambda_x[..., 1], glambda_x[..., 0], cm)
print('--------------------------')
