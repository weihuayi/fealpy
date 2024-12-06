
import sympy as sp

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
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

    
    def grad_basis(self, p, mi):
        phi = self.basis(p, mi)  # 获取基函数
        ldof = len(phi)
        grad_phi = []
        
        # 对每个基函数计算导数
        for i in range(ldof):
            # 计算对每个重心坐标的导数
            grad_i = [sp.diff(phi[i], self.l[j]) for j in range(self.GD + 1)]
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
                        temp = gphi1[i][m] * gphi2[j][n]
                        S[i, j, m, n] = self.integrate(temp)
                    
        return S

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
