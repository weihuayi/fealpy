"""Method of Moving Asymptotes (MMA) 子问题求解器

该模块实现了 MMA 算法中子问题的求解,使用原始-对偶内点法.
参考：Svanberg, K. (1987). The method of moving asymptotes—a new method for 
structural optimization. International Journal for Numerical Methods in Engineering.
"""

from typing import Tuple
from numpy.linalg import solve

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.solver import cg, spsolve

def solve_mma_subproblem(m: int, n: int, 
                        epsimin: float,
                        low: TensorLike, upp: TensorLike,
                        alfa: TensorLike, beta: TensorLike,
                        p0: TensorLike, q0: TensorLike,
                        P: TensorLike, Q: TensorLike,
                        a0: float,
                        a: TensorLike = None,
                        b: TensorLike = None,
                        c: TensorLike = None,
                        d: TensorLike = None) -> Tuple[TensorLike, ...]:
    """求解 MMA 子问题
    
    使用 primal-dual Newton method 求解由 MMA 算法产生的凸近似子问题.
    
    Parameters
    ----------
    m : 约束数量, 约束函数 f_i(x) 的个数
    n : 设计变量数量, 变量 x_j 的个数
    epsimin : 最小收敛容差
    low : 下渐近线
    upp : 上渐近线
    alfa : 变量下界
    beta : 变量上界
    p0 (n, 1): 目标函数的正梯度项
    q0 (n, 1): 目标函数的负梯度项
    P (m, n): 约束函数的正梯度项
    Q (m, n): 约束函数的负梯度项
    a0 (float): 目标函数的线性项 a_0*z 的系数
    a (m, 1): 约束的线性项 a_i*z 的系数
    b (m, 1): -r_i
    c (m, 1): 约束的二次项 c_iy_i 的系数
    d (m, 1): 约束的二次项 0.5*d_i*y_i^2 的系数
    
    Returns
    - xmma (n, 1): 自然变量
    - ymma: 人工变量 
    - zmma: 人工变量
    - lam (m, 1): m 个约束的拉格朗日乘子 lambda
    - xsi (n, 1): n 个约束 alpha_j-x_j 的拉格朗日乘子 xi
    - eta : n 个约束 x_j-beta_j 的拉格朗日乘子 eta
    - mu : m 个约束 -y_i <= 0 的拉格朗日乘子 mu
    - zet : 1 个约束 -z <= 0 的拉格朗日乘子 zeta
    - s : m 个约束的松弛变量 s
    """

    # 变量初始化
    een = bm.ones((n, 1))
    eem = bm.ones((m, 1))
    x = 0.5 * (alfa + beta)
    y = bm.copy(eem)
    z = bm.array([[1.0]])
    lam = bm.copy(eem)
    xsi = een / (x - alfa)
    xsi = bm.maximum(xsi, een)
    eta = een / (beta - x)
    eta = bm.maximum(eta, een)
    mu = bm.maximum(eem, 0.5*c)
    zet = bm.array([[1.0]])
    s = bm.copy(eem)

    epsi = 1 # 松弛参数, 每次外循环迭代中逐步减小   
    epsvecn = epsi * een # (m, 1)
    epsvecm = epsi * eem # (n, 1)
    # 外循环迭代: 逐步减小松弛参数 epsi
    itera = 0
    while epsi > epsimin:
        epsvecn = epsi * een # (m, 1)
        epsvecm = epsi * eem # (n, 1)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + bm.dot(P.T, lam)
        qlam = q0 + bm.dot(Q.T, lam)
        gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2

        # 1. 计算 KKT 残差
        rex = dpsidx - xsi + eta           # (n, 1)
        rey = c + d*y - mu - lam           # (m, 1) 
        rez = a0 - zet - bm.dot(a.T, lam)  # (m, 1)
        relam = gvec - a*z - y + s - b     # (m, 1)
        rexsi = xsi * (x - alfa) - epsvecn # (n, 1)
        reeta = eta * (beta - x) - epsvecn # (n, 1)
        remu = mu * y - epsvecm            # (m, 1)
        rezet = zet * z - epsi             # (1, 1)
        res = lam * s - epsvecm            # (m, 1)
    
        residu1 = bm.concatenate((rex, rey, rez), axis=0)   # (n+m+m, 1)
        residu2 = bm.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0) # (m+n+n+m+1+m, 1)
        residu = bm.concatenate((residu1, residu2), axis=0)
        residunorm = bm.sqrt((bm.dot(residu.T, residu)).item())
        residumax = bm.max(bm.abs(residu))
        
        ittt = 0
        # 内循环迭代: 在固定的 epsi 下求解 KKT 系统
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt = ittt + 1
            itera = itera + 1
            
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2 # (n, 1)
            xlinv2 = een / xl2 # (n, 1)
            
            plam = p0 + bm.dot(P.T, lam)
            qlam = q0 + bm.dot(Q.T, lam)
            gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)

            # TODO 使用 einsum 替代对角矩阵乘法
            GG = bm.einsum('j, ij -> ij', uxinv2.flatten(), P) - \
                 bm.einsum('j, ij -> ij', xlinv2.flatten(), Q)  # (m, n) 
            # GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - \
            #      (diags(xlinv2.flatten(), 0).dot(Q.T)).T # (m, n)
            
            # 2. 计算 Newton 方向的一阶残差 delta_x, delta_y, delta_z, delta_lambda
            dpsidx = plam / ux2 - qlam / xl2                            # (n, 1)
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x) # (n, 1)
            dely = c + d * y - lam - epsvecm / y                        # (m, 1)
            delz = a0 - bm.dot(a.T, lam) - epsi / z                     # (1, 1)
            dellam = gvec - a * z - y - b + epsvecm / lam               # (m, 1)
            
            # 3. 计算 Hessian 的对角线
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x) # (n, 1)
            diagxinv = een / diagx
            diagy = d + mu / y                                      # (m, 1)
            diagyinv = eem/diagy
            diaglam = s / lam                                       # (m, 1)
            diaglamyi = diaglam + diagyinv
            
            # 4. 求解 KKT 线性系统
            if m < n:
                # 选择 (\Delta\lambda, \Delta z) 系统
                blam = dellam + dely / diagy - bm.dot(GG, (delx / diagx)) # (m, 1)
                bb = bm.concatenate((blam, delz), axis=0)                 # (m+1, 1)
                # TODO 使用 einsum 替代对角矩阵乘法
                D_lamyi = diaglamyi * bm.eye(1)  
                GD_xG = bm.einsum('ik, k, jk -> ij', GG, diagxinv.flatten(), GG)  
                Alam = D_lamyi + GD_xG  # (m, 1)
                # Alam = bm.asarray(diags(diaglamyi.flatten(), 0) + \
                #         (diags(diagxinv.flatten(), 0).dot(GG.T).T).dot(GG.T))
                AAr1 = bm.concatenate((Alam, a), axis=1)     # (m, m+1)
                AAr2 = bm.concatenate((a, -zet/z), axis=0).T # (1, m+1)
                AA = bm.concatenate((AAr1, AAr2), axis=0)    # (m+1, m+1)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]                                # (m, 1)
                dx = -delx / diagx - bm.dot(GG.T, dlam) / diagx  # (m, 1)
            else:
                # 选择 (\Delta x, \Delta z) 系统
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely/diagy
                # TODO 使用 einsum 替代对角矩阵乘法
                D_x = diagx * bm.eye(1)
                GD_lamyiG = bm.einsum('ik, k, jk -> ij', GG, diaglamyiinv.flatten(), GG)
                Axx = D_x + GD_lamyiG
                # Axx = bm.asarray(diags(diagx.flatten(), 0) + \
                #     (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG))
                azz = zet/z + bm.dot(a.T, (a/diaglamyi))
                axz = bm.dot(-GG.T, (a/diaglamyi))
                bx = delx + bm.dot(GG.T, (dellamyi/diaglamyi))
                bz = delz - bm.dot(a.T, (dellamyi/diaglamyi))
                AAr1 = bm.concatenate((Axx, axz), axis=1)
                AAr2 = bm.concatenate((axz.T, azz), axis=1)
                AA = bm.concatenate((AAr1, AAr2), axis=0) # (n+1, n+1)
                bb = bm.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n+1]                                                               # (m, 1)
                dlam = bm.dot(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi # (m, 1)
                
            # 5.计算 Newton 方向 \Delta w
            dy = -dely / diagy + dlam / diagy                  # (m, 1)
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa) # (n, 1)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x) # (n, 1)
            dmu = -mu + epsvecm / y - (mu * dy) / y            # (m, 1)
            dzet = -zet + epsi / z - zet * dz / z              # (1, 1)
            ds = -s + epsvecm / lam - (s * dlam) / lam         # (m, 1)
            xx = bm.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)          # (m+n+n+m+1+m, 1)
            dxx = bm.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0) # (m+n+n+m+1+m, 1)
            
            # 6. 确定线搜索步长
            stepxx = -1.01 * dxx / xx
            stmxx = bm.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = bm.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = bm.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv
            
            xold = bm.copy(x)
            yold = bm.copy(y)
            zold = bm.copy(z)
            lamold = bm.copy(lam)
            xsiold = bm.copy(xsi)
            etaold = bm.copy(eta)
            muold = bm.copy(mu)
            zetold = bm.copy(zet)
            sold = bm.copy(s)
            
            # 7. 线搜索更新变量
            itto = 0
            resinew = 2 * residunorm
            while (resinew > residunorm) and (itto < 50):
                itto = itto + 1
 
                x = xold + steg*dx
                y = yold + steg*dy
                z = zold + steg*dz
                lam = lamold + steg*dlam
                xsi = xsiold + steg*dxsi
                eta = etaold + steg*deta
                mu = muold + steg*dmu
                zet = zetold + steg*dzet
                s = sold + steg*ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1

                plam = p0 + bm.dot(P.T, lam)
                qlam = q0 + bm.dot(Q.T, lam)
                gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
                dpsidx = plam/ux2 - qlam/xl2

                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - bm.dot(a.T, lam)
                relam = gvec - a*z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = bm.concatenate((rex, rey, rez), axis=0)
                residu2 = bm.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = bm.concatenate((residu1, residu2), axis=0)
                resinew = bm.sqrt(bm.dot(residu.T, residu))

                steg = steg / 2

            residunorm = resinew.copy()
            residumax = bm.max(bm.abs(residu))
            steg = 2 * steg
            
        epsi = 0.1 * epsi
    
    # 返回最优解
    xmma = bm.copy(x)
    ymma = bm.copy(y)
    zmma = bm.copy(z)
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma