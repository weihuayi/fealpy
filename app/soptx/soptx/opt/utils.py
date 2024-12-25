"""Method of Moving Asymptotes (MMA) 子问题求解器

该模块实现了 MMA 算法中子问题的求解,使用原始-对偶内点法.
参考：Svanberg, K. (1987). The method of moving asymptotes—a new method for 
structural optimization. International Journal for Numerical Methods in Engineering.
"""

from typing import Tuple
from numpy import diag as diags
from numpy.linalg import solve

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

def solve_mma_subproblem(m: int, 
                        n: int, 
                        epsimin: float,
                        low: TensorLike,
                        upp: TensorLike,
                        alfa: TensorLike,
                        beta: TensorLike,
                        p0: TensorLike,
                        q0: TensorLike,
                        P: TensorLike,
                        Q: TensorLike,
                        a0: float,
                        a: TensorLike = None,
                        b: TensorLike = None,
                        c: TensorLike = None,
                        d: TensorLike = None) -> Tuple[TensorLike, ...]:
    """求解 MMA 子问题
    
    使用原始-对偶内点法求解由 MMA 算法产生的凸近似子问题.
    
    Parameters
    ----------
    m : 约束数量
    n : 设计变量数量
    epsimin : 最小收敛容差
    low : 下渐近线
    upp : 上渐近线
    alfa : 变量下界
    beta : 变量上界
    p0 : 目标函数的正梯度项
    q0 : 目标函数的负梯度项
    P : 约束函数的正梯度项
    Q : 约束函数的负梯度项
    a0 : 弹性权重
    a : 约束线性项系数
    b : 约束常数项
    c : 约束二次项系数
    d : 约束二次项系数
    
    Returns
    -------
    xmma : 最优设计变量
    ymma : 对偶变量
    zmma : 松弛变量
    lam : 拉格朗日乘子
    xsi : 对偶变量
    eta : 对偶变量
    mu : 对偶变量
    zet : 对偶变量
    s : 松弛变量
    """

    een = bm.ones((n, 1))
    eem = bm.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
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
    itera = 0

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        # 计算梯度和残差
        plam = p0 + bm.dot(P.T, lam)
        qlam = q0 + bm.dot(Q.T, lam)
        gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
        dpsidx = plam/ux2 - qlam/xl2
        rex = dpsidx - xsi + eta
        rey = c + d*y - mu - lam
        rez = a0 - zet - bm.dot(a.T, lam)
        relam = gvec - a*z - y + s - b
        rexsi = xsi*(x-alfa) - epsvecn
        reeta = eta*(beta-x) - epsvecn
        remu = mu*y - epsvecm
        rezet = zet*z - epsi
        res = lam*s - epsvecm
        
        residu1 = bm.concatenate((rex, rey, rez), axis=0)
        residu2 = bm.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = bm.concatenate((residu1, residu2), axis=0)
        residunorm = bm.sqrt((bm.dot(residu.T, residu)).item())
        residumax = bm.max(bm.abs(residu))
        
        ittt = 0
        # 内循环迭代求解 KKT 系统
        while (residumax > 0.9*epsi) and (ittt < 200):
            ittt = ittt + 1
            itera = itera + 1
            
            # 计算中间变量
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl1
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            
            # 计算梯度矩阵和向量
            plam = p0 + bm.dot(P.T, lam)
            qlam = q0 + bm.dot(Q.T, lam)
            gvec = bm.dot(P, uxinv1) + bm.dot(Q, xlinv1)
            GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - \
                 (diags(xlinv2.flatten(), 0).dot(Q.T)).T
            
            # 计算残差
            dpsidx = plam/ux2 - qlam/xl2
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - bm.dot(a.T, lam) - epsi/z
            dellam = gvec - a*z - y - b + epsvecm/lam
            
            # 计算 Hessian 对角线
            diagx = plam/ux3 + qlam/xl3
            diagx = 2*diagx + xsi/(x-alfa) + eta/(beta-x)
            diagxinv = een/diagx
            diagy = d + mu/y
            diagyinv = eem/diagy
            diaglam = s/lam
            diaglamyi = diaglam + diagyinv
            
            # 求解线性系统
            if m < n:
                blam = dellam + dely / diagy - bm.dot(GG, (delx / diagx))
                bb = bm.concatenate((blam, delz), axis=0)
                Alam = bm.asarray(diags(diaglamyi.flatten(), 0) + \
                        (diags(diagxinv.flatten(), 0).dot(GG.T).T).dot(GG.T))
                AAr1 = bm.concatenate((Alam, a), axis=1)
                AAr2 = bm.concatenate((a, -zet/z), axis=0).T
                AA = bm.concatenate((AAr1, AAr2), axis=0)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx / diagx - bm.dot(GG.T, dlam) / diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam + dely/diagy
                Axx = bm.asarray(diags(diagx.flatten(), 0) + \
                    (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG))
                azz = zet/z + bm.dot(a.T, (a/diaglamyi))
                axz = bm.dot(-GG.T, (a/diaglamyi))
                bx = delx + bm.dot(GG.T, (dellamyi/diaglamyi))
                bz = delz - bm.dot(a.T, (dellamyi/diaglamyi))
                AAr1 = bm.concatenate((Axx, axz), axis=1)
                AAr2 = bm.concatenate((axz.T, azz), axis=1)
                AA = bm.concatenate((AAr1, AAr2), axis=0)
                bb = bm.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = bm.dot(GG, dx)/diaglamyi - dz*(a/diaglamyi) + \
                       dellamyi/diaglamyi
                
            # 计算其他变量的更新
            dy = -dely/diagy + dlam/diagy
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet*dz/z
            ds = -s + epsvecm/lam - (s*dlam)/lam
            xx = bm.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = bm.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)
            
            # 步长确定
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
            
            # 保存旧值
            xold = bm.copy(x)
            yold = bm.copy(y)
            zold = bm.copy(z)
            lamold = bm.copy(lam)
            xsiold = bm.copy(xsi)
            etaold = bm.copy(eta)
            muold = bm.copy(mu)
            zetold = bm.copy(zet)
            sold = bm.copy(s)
            
            # 线搜索
            itto = 0
            resinew = 2*residunorm
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
                rexsi = xsi*(x-alfa) - epsvecn
                reeta = eta*(beta-x) - epsvecn
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