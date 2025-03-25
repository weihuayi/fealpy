
import sympy as sp

from typing import Optional, List

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.functionspace import FunctionSpace, LagrangeFESpace
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator
from fealpy.mesh import SimplexMesh, TensorMesh, UniformMesh2d, UniformMesh3d

class LinearSymbolicIntegration:
    def __init__(self, space1: FunctionSpace, space2 : Optional[FunctionSpace]=None):
        """
        初始化符号积分类
        
        Parameters
        - space1 : 第一个有限元空间
        - space2 : 第二个有限元空间，默认与第一个相同
        """
        self.space1 = space1
        self.mesh = space1.mesh
        self.p1 = space1.p  # 第一个空间的多项式次数
        self.GD = self.mesh.geo_dimension()  # 几何维度
        self.ldof1 = space1.number_of_local_dofs()  # 第一个空间的局部自由度数量

        # 如果没有提供第二个空间，则假设两个空间相同
        if space2 is None:
            self.space2 = space1
            self.p2 = self.p1
            self.ldof2 = self.ldof1
        else:
            self.space2 = space2
            self.p2 = space2.p  # 第二个空间的多项式次数
            self.ldof2 = space2.number_of_local_dofs()  # 第二个空间的局部自由度数量

        # 根据单元类型设置
        if isinstance(self.mesh, SimplexMesh):
            # 对于单纯形, 使用重心坐标 λ_i
            self.l = sp.symbols('l0:%d' % (self.GD+1), real=True)
            self.mi1 = self.mesh.multi_index_matrix(p=self.p1, etype=self.GD)
            if space2 is None:
                self.mi2 = self.mi1
            else:
                self.mi2 = self.mesh.multi_index_matrix(p=self.p2, etype=self.GD) # 第二个空间的多重指标矩阵
        elif isinstance(self.mesh, UniformMesh2d):
            # 对于二维结构网格, 设置参考坐标
            self.setup_structured_2d_symbols()
        else:
            # 对于三维结构网格, 设置参考坐标
            self.setup_structured_3d_symbols()

    def setup_structured_2d_symbols(self) -> None:
        """设置二维结构网格的参考坐标(使用 [0,1]² 参考单元)"""
        # 参考坐标
        self.xi, self.eta = sp.symbols('xi eta', real=True)
        
        # 双线性形函数 (在 [0, 1]² 上), 先 y 后 x 的顺序
        self.N = [
            (1 - self.xi) * (1 - self.eta), # (0, 0) 节点
            (1 - self.xi) * self.eta,       # (0, 1) 节点
            self.xi * (1 - self.eta),       # (1, 0) 节点
            self.xi * self.eta              # (1, 1) 节点
        ]

    def setup_structured_3d_symbols(self) -> None:
        """设置三维结构网格的参考坐标 (使用 [0,1]³ 参考单元)"""
        # 参考坐标
        self.xi, self.eta, self.zeta = sp.symbols('xi eta zeta', real=True)
        
        # 三线性形函数 (在[0,1]³上), 先 z 再 y 最后 x 的顺序
        self.N = [
            (1 - self.xi) * (1 - self.eta) * (1 - self.zeta),  # (0, 0, 0)
            (1 - self.xi) * (1 - self.eta) * self.zeta,        # (0, 0, 1)
            (1 - self.xi) * self.eta * (1 - self.zeta),        # (0, 1, 0)
            (1 - self.xi) * self.eta * self.zeta,              # (0, 1, 1)
            self.xi * (1 - self.eta) * (1 - self.zeta),        # (1, 0, 0)
            self.xi * (1 - self.eta) * self.zeta,              # (1, 0, 1)
            self.xi * self.eta * (1 - self.zeta),              # (1, 1, 0)
            self.xi * self.eta * self.zeta                     # (1, 1, 1)
        ]
    
    def compute_mapping(self, vertices: TensorLike) -> sp.tensor.array.MutableDenseNDimArray:
        """计算单元的映射"""
        if isinstance(self.mesh, UniformMesh2d):
            return self.compute_structured_2d_mapping(vertices)
        else:
            return self.compute_structured_3d_mapping(vertices)

    def compute_structured_2d_mapping(self, vertices):
        """计算二维结构单元的映射"""
        # NC = vertices.shape[0]
        # NCN = vertices.shape[1]

        # JG_inv_array = sp.tensor.array.MutableDenseNDimArray(
        #                                     sp.zeros(NC * self.GD * self.GD),
        #                                     (NC, self.GD, self.GD)
        #                                 )
        
        # # 对每个单元计算 JG^(-1) 符号表达式
        # for cell_idx in range(NC):
        #     # 提取单元顶点
        #     cell_vertices = vertices[cell_idx]
        #     x_coords = cell_vertices[:, 0]
        #     y_coords = cell_vertices[:, 1]
            
        #     # 构建映射函数
        #     x_map = sum(self.N[i] * float(x_coords[i]) for i in range(NCN))
        #     y_map = sum(self.N[i] * float(y_coords[i]) for i in range(NCN))
            
        #     J = sp.Matrix([
        #         [sp.diff(x_map, self.xi), sp.diff(x_map, self.eta)],
        #         [sp.diff(y_map, self.xi), sp.diff(y_map, self.eta)]
        #     ])
        #     # 逐元素简化
        #     for i in range(J.shape[0]):
        #         for j in range(J.shape[1]):
        #             if J[i, j] != 0:  
        #                 J[i, j] = sp.expand(J[i, j])
        #                 J[i, j] = sp.simplify(J[i, j])
        #     # 计算第一基本形式 G = J^T·J
        #     G = J.transpose() * J
        #     G_inv = G.inv()
        #     JG_inv = J * G_inv
            
        #     for i in range(self.GD):
        #         for j in range(self.GD):
        #             JG_inv_array[cell_idx, i, j] = JG_inv[i, j]

        NC = vertices.shape[0]
        JG_inv_array = bm.zeros((NC, self.GD, self.GD), dtype=bm.float64, device=self.mesh.device)
        
        inv_h = bm.array([1/h for h in self.mesh.h])
        i = bm.arange(self.GD)
        JG_inv_array = bm.set_at(JG_inv_array, (..., i, i), inv_h)
        
        return JG_inv_array
    
    def compute_structured_3d_mapping(self, vertices: TensorLike) -> sp.tensor.array.MutableDenseNDimArray:
        """计算三维结构单元的映射"""
        # NC = vertices.shape[0]
        # NCN = vertices.shape[1]
        
        # JG_inv_array = sp.tensor.array.MutableDenseNDimArray(
        #                                     sp.zeros(NC * self.GD * self.GD),
        #                                     (NC, self.GD, self.GD)
        #                                 )
        
        # for cell_idx in range(NC):
        #     # 提取单元顶点
        #     cell_vertices = vertices[cell_idx]
        #     x_coords = cell_vertices[:, 0]
        #     y_coords = cell_vertices[:, 1]
        #     z_coords = cell_vertices[:, 2]
        #     # 映射函数
        #     x_map = sum(self.N[i] * float(x_coords[i]) for i in range(NCN))
        #     y_map = sum(self.N[i] * float(y_coords[i]) for i in range(NCN))
        #     z_map = sum(self.N[i] * float(z_coords[i]) for i in range(NCN))
        #     # 雅可比矩阵
        #     J = sp.Matrix([
        #         [sp.diff(x_map, self.xi), sp.diff(x_map, self.eta), sp.diff(x_map, self.zeta)],
        #         [sp.diff(y_map, self.xi), sp.diff(y_map, self.eta), sp.diff(y_map, self.zeta)],
        #         [sp.diff(z_map, self.xi), sp.diff(z_map, self.eta), sp.diff(z_map, self.zeta)]
        #     ])
        #     # 逐元素简化
        #     for i in range(J.shape[0]):
        #         for j in range(J.shape[1]):
        #             if J[i,j] != 0:  
        #                 J[i,j] = sp.expand(J[i,j])
        #                 J[i,j] = sp.simplify(J[i,j])
        #     # 计算第一基本形式 G = J^T·J
        #     G = J.transpose() * J
        #     # 计算 G 的逆矩阵
        #     G_inv = G.inv()
        #     # 计算 JG^(-1), 表示参考坐标关于物理坐标的导数
        #     JG_inv = J * G_inv
        #     # 存储结果
        #     for i in range(self.GD):
        #         for j in range(self.GD):
        #             JG_inv_array[cell_idx, i, j] = JG_inv[i, j]

        NC = vertices.shape[0]
        JG_inv_array = bm.zeros((NC, self.GD, self.GD), dtype=bm.float64, device=self.mesh.device)
        
        inv_h = bm.array([1/h for h in self.mesh.h])
        i = bm.arange(self.GD)
        JG_inv_array = bm.set_at(JG_inv_array, (..., i, i), inv_h)

        return JG_inv_array

    def basis(self, p: int, mi: Optional[TensorLike]=None) -> List[sp.Expr]:
        """
        根据单元类型计算基函数

        Parameters
        - p : 多项式次数
        - mi : 多重指标矩阵, 默认为 None
        
        Returns
        - 基函数列表
        """
        if isinstance(self.mesh, SimplexMesh):
            return self.basis_simplex(p, mi)
        elif isinstance(self.mesh, UniformMesh2d):
            return self.basis_structured_2d(p)
        else:
            return self.basis_structured_3d(p)

    def basis_simplex(self, p: int, mi: TensorLike) -> List[sp.Expr]:
        """计算 p 次拉格朗日基函数
        
        使用递推方法构造基函数, 可以得到正确的系数和形式
        例如, 对于 p = 3 时：
        - 顶点基函数: λᵢ(3λᵢ - 2)(3λᵢ - 1)/4
        - 边内部点基函数: 9λᵢλⱼ(3λᵢ - 1)/2
        - 内部点基函数: 27λ₀λ₁λ₂
        
        Parameters
        - p : 多项式次数
        - mi : 多重指标矩阵, 形状为 (ldof, GD+1)
            
        Returns
        - phi : 基函数列表
        """
        GD = self.GD
        ldof = mi.shape[0]
        l = self.l
        
        # 初始化矩阵 A, 大小为 (p+1) x (GD+1)
        A = sp.ones(p+1, GD+1)
        
        # 递推计算 A 矩阵的元素
        for i in range(1, p+1):
            for j in range(GD+1):
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
    
    def basis_structured_2d(self, p: int) -> List[sp.Expr]:
        """计算二维结构网格的基函数 (使用张量积 Lagrange 多项式)"""
        # 在 [0,1] 上创建等距点
        xi_points = [i/p for i in range(p+1)]
        eta_points = bm.copy(xi_points)
        
        # 计算一维Lagrange多项式
        L_xi = []
        for i in range(p+1):
            L_i = 1
            for j in range(p+1):
                if i != j:
                    L_i *= (self.xi - xi_points[j]) / (xi_points[i] - xi_points[j])
            L_xi.append(L_i)
        
        L_eta = []
        for i in range(p+1):
            L_i = 1
            for j in range(p+1):
                if i != j:
                    L_i *= (self.eta - eta_points[j]) / (eta_points[i] - eta_points[j])
            L_eta.append(L_i)
        
        # 创建张量积基函数, 顺序是先 η 再 ξ
        phi = []
        for j in range(p+1):
            for i in range(p+1):
                phi.append(L_xi[i] * L_eta[j])
        
        return phi

    def basis_structured_3d(self, p: int) -> List[sp.Expr]:
        """计算三维结构网格的基函数 (使用张量积 Lagrange 多项式)"""
        # 在 [0,1] 上创建等距点
        xi_points = [i/p for i in range(p+1)]
        eta_points = bm.copy(xi_points)
        zeta_points = bm.copy(xi_points)
        
        # 计算一维 Lagrange 多项式
        L_xi = []
        for i in range(p+1):
            L_i = 1
            for j in range(p+1):
                if i != j:
                    L_i *= (self.xi - xi_points[j]) / (xi_points[i] - xi_points[j])
            L_xi.append(L_i)
        
        L_eta = []
        for i in range(p+1):
            L_i = 1
            for j in range(p+1):
                if i != j:
                    L_i *= (self.eta - eta_points[j]) / (eta_points[i] - eta_points[j])
            L_eta.append(L_i)
        
        L_zeta = []
        for i in range(p+1):
            L_i = 1
            for j in range(p+1):
                if i != j:
                    L_i *= (self.zeta - zeta_points[j]) / (zeta_points[i] - zeta_points[j])
            L_zeta.append(L_i)
        
        # 创建张量积基函数, 顺序是先 zeta 再 η 最后 ξ
        phi = []
        for k in range(p+1):
            for j in range(p+1):
                for i in range(p+1):
                    phi.append(L_xi[i] * L_eta[j] * L_zeta[k])
        
        return phi
    
    def grad_basis(self, p: int, mi: Optional[TensorLike]=None) -> List[List[sp.Expr]]:
        """
        计算基函数的梯度

        Parameters
        - p : 多项式次数
        - mi : 多重指标矩阵, 默认为 None
        
        Returns
        - grad_phi : 基函数梯度列表, 每个元素是一个列表, 包含该基函数对各坐标的偏导数
        """
        if isinstance(self.mesh, SimplexMesh):
            phi = self.basis(p, mi)
            # 对单纯形，导数是相对于重心坐标
            vars_list = self.l[:self.GD + 1]
        elif isinstance(self.mesh, UniformMesh2d):
            phi = self.basis(p)
            # 对四边形，导数是相对于参考坐标
            vars_list = [self.eta, self.xi]
        else:
            phi = self.basis(p)
            # 对六面体，导数是相对于参考坐标
            vars_list = [self.zeta, self.eta, self.xi]
        
        ldof = len(phi)
        grad_phi = []
        
        # 对每个基函数计算导数
        for i in range(ldof):
            grad_i = [sp.diff(phi[i], var) for var in vars_list]
            grad_phi.append(grad_i)
        
        return grad_phi
    
    def multi_index(self, monomial: sp.Expr) -> TensorLike:
        l = self.l
        GD = self.GD

        m = monomial.as_powers_dict()
        a = bm.zeros(GD+1, dtype=bm.int32) # 返回幂指标
        for i in range(GD+1):
            a[i] = int(m.get(l[i]) or 0)

        return a
    
    def integrate(self, f: sp.Expr) -> sp.Expr:
        """积分函数"""
        if isinstance(self.mesh, SimplexMesh):
            return self.integrate_simplex(f)
        elif isinstance(self.mesh, UniformMesh2d):
            return self.integrate_structured_2d(f)
        else:
            return self.integrate_structured_3d(f)
    
    def integrate_simplex(self, f: sp.Expr) -> sp.Expr:
        """在单纯形上积分"""
        GD = self.GD
        
        f = f.expand()
        r = 0    # 积分值
        
        # 处理非常数项，完全按照原始实现
        for m in f.as_coeff_add()[1]:
            c = m.as_coeff_mul()[0]   # 返回系数
            a = self.multi_index(m)   # 返回单项式的幂指标
            temp = 1
            for i in range(GD+1):
                temp *= sp.factorial(a[i])
            r += sp.factorial(GD) * c * temp / sp.factorial(sum(a) + GD)
        
        # 处理常数项
        return r + f.as_coeff_add()[0]
    
    def integrate_structured_2d(self, f: sp.Expr) -> sp.Expr:
        """在参考二维结构单元上积分"""
        try:
            result = sp.integrate(sp.integrate(f, 
                                               (self.xi, 0, 1)), 
                                               (self.eta, 0, 1))
            return result
        except Exception as e:
            error_msg = f"符号积分失败: {str(e)}. 此积分可能不具有解析解, 请考虑使用数值积分方法."
            print(error_msg)  # 打印错误信息
            raise ValueError(error_msg)  # 抛出更具体的错误

    def integrate_structured_3d(self, f: sp.Expr) -> sp.Expr:
        """在参考三维结构单元上积分"""
        try:
            result = sp.integrate(sp.integrate(sp.integrate(f, 
                                                            (self.xi, 0, 1)), 
                                                            (self.eta, 0, 1)), 
                                                            (self.zeta, 0, 1))
            return result
        except Exception as e:
            error_msg = f"符号积分失败: {str(e)}. 此积分可能不具有解析解, 请考虑使用数值积分方法."
            print(error_msg)  # 打印错误信息
            raise ValueError(error_msg)  # 抛出更具体的错误
    
    def phi_phi_matrix(self) -> sp.tensor.array.MutableDenseNDimArray:
        if isinstance(self.mesh, SimplexMesh):
            phi1 = self.basis(self.p1, self.mi1)
            phi2 = self.basis(self.p2, self.mi2)
        else:
            phi1 = self.basis(self.p1)
            phi2 = self.basis(self.p2)
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
    
    def gphi_gphi_matrix(self) -> sp.tensor.array.MutableDenseNDimArray:
        """
        计算基函数梯度在参考坐标系下的积分张量
        
        Returns
        - S : 基函数梯度积分张量, 形状为 (ldof1, ldof2, dim, dim)
              S[i, j, m, n] = ∫ (∂φ₁ᵢ/∂ξₘ · ∂φ₂ⱼ/∂ξₙ) dξ
        """
        if isinstance(self.mesh, SimplexMesh):
            gphi1 = self.grad_basis(self.p1, self.mi1)
            gphi2 = self.grad_basis(self.p2, self.mi2)
            dim = self.GD + 1  # 重心坐标的数量
        else:
            gphi1 = self.grad_basis(self.p1)
            gphi2 = self.grad_basis(self.p2)
            dim = self.GD      # 参考坐标的数量 (ξ, η) 或 (ξ, η, ζ)

        S = sp.tensor.array.MutableDenseNDimArray(
                                sp.zeros(self.ldof1 * self.ldof2 * dim * dim),
                                (self.ldof1, self.ldof2, dim, dim)
                            )
        
        # for i in range(self.ldof1):
        #     for j in range(self.ldof2):
        #         for m in range(dim):
        #             for n in range(dim):
        #                 integrand = gphi1[i][m] * gphi2[j][n]
        #                 S[i, j, m, n] = self.integrate(integrand)

        integral_cache = {}  # 缓存已计算的积分结果
        
        for i in range(self.ldof1):
            for j in range(self.ldof2):
                for m in range(dim):
                    for n in range(dim):
                        # 构建积分表达式的哈希键
                        expr = gphi1[i][m] * gphi2[j][n]
                        expr_key = str(expr)
                        
                        # 检查缓存中是否已有结果
                        if expr_key in integral_cache:
                            S[i, j, m, n] = integral_cache[expr_key]
                        else:
                            result = self.integrate(expr)
                            integral_cache[expr_key] = result
                            S[i, j, m, n] = result

        return S

class NonlinearSymbolicIntegration:
    pass

def normal_strain(gphi: TensorLike, indices: TensorLike, *, out:
                  Optional[TensorLike]=None) -> TensorLike:
    """Assembly normal strain tensor.

    Parameters:
        gphi (TensorLike): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (TensorLike | None, optional): Output tensor. Defaults to None.

    Returns:
        TensorLike: Normal strain shaped (..., GD, GD*ldof).
    """
    kwargs = bm.context(gphi)
    ldof, GD = gphi.shape[-2:]
    new_shape = gphi.shape[:-2] + (GD, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    for i in range(GD):
        out = bm.set_at(out, (..., i, indices[:, i]), gphi[..., :, i])
        # out[..., i, indices[:, i]] = gphi[..., :, i]

    return out


def shear_strain(gphi: TensorLike, indices: TensorLike, *, out:
                 Optional[TensorLike]=None) -> TensorLike:
    """Assembly shear strain tensor.

    Parameters:
        gphi (TensorLike): Gradient of the scalar basis functions shaped (..., ldof, GD).\n
        indices (bool, optional): Indices of DoF components in the flattened DoF, shaped (ldof, GD).\n
        out (TensorLike | None, optional): Output tensor. Defaults to None.

    Returns:
        TensorLike: Sheared strain shaped (..., NNZ, GD*ldof) where NNZ = (GD + (GD+1))//2.
    """
    kwargs = bm.context(gphi)
    ldof, GD = gphi.shape[-2:]
    if GD < 2:
        raise ValueError(f"The shear strain requires GD >= 2, but GD = {GD}")
    NNZ = (GD * (GD-1))//2
    new_shape = gphi.shape[:-2] + (NNZ, GD*ldof)

    if out is None:
        out = bm.zeros(new_shape, **kwargs)
    else:
        if out.shape != new_shape:
            raise ValueError(f'out.shape={out.shape} != {new_shape}')

    cursor = 0

    for i in range(0, GD-1):
        for j in range(i+1, GD):
            out = bm.set_at(out, (..., cursor, indices[:, i]), gphi[..., :, j])
            out = bm.set_at(out, (..., cursor, indices[:, j]), gphi[..., :, i])
            cursor += 1

    return out