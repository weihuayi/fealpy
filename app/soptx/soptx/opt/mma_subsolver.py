"""Method of Moving Asymptotes (MMA) 子问题求解器

该模块实现了 MMA 算法中子问题的求解,使用原始-对偶内点法。
参考：Svanberg, K. (1987). The method of moving asymptotes—a new method for 
structural optimization. International Journal for Numerical Methods in Engineering.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve
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
    
    使用原始-对偶内点法求解由 MMA 算法产生的凸近似子问题。
    
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
    if a is None:
        a = np.zeros((m, 1))
    if b is None:
        b = np.zeros((m, 1))
    if c is None:
        c = 1000*np.ones((m, 1))
    if d is None:
        d = np.zeros((m, 1))

    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi*een
    epsvecm = epsi*eem
    x = 0.5*(alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een/(x-alfa)
    xsi = np.maximum(xsi, een)
    eta = een/(beta-x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5*c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0

    # 主循环,直到障碍参数小于容差
    while epsi > epsimin:
        epsvecn = epsi*een
        epsvecm = epsi*eem
        ux1 = upp-x
        xl1 = x-low
        ux2 = ux1*ux1
        xl2 = xl1*xl1
        uxinv1 = een/ux1
        xlinv1 = een/xl1

        # 计算梯度和残差
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
        dpsidx = plam/ux2 - qlam/xl2
        rex = dpsidx - xsi + eta
        rey = c + d*y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - a*z - y + s - b
        rexsi = xsi*(x-alfa) - epsvecn
        reeta = eta*(beta-x) - epsvecn
        remu = mu*y - epsvecm
        rezet = zet*z - epsi
        res = lam*s - epsvecm
        
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt((np.dot(residu.T, residu)).item())
        residumax = np.max(np.abs(residu))
        
        ittt = 0
        # 内循环迭代求解 KKT 系统
        while (residumax > 0.9*epsi) and (ittt < 200):
            ittt = ittt + 1
            itera = itera + 1
            
            # 计算中间变量
            ux1 = upp-x
            xl1 = x-low
            ux2 = ux1*ux1
            xl2 = xl1*xl2
            ux3 = ux1*ux2
            xl3 = xl1*xl2
            uxinv1 = een/ux1
            xlinv1 = een/xl1
            uxinv2 = een/ux2
            xlinv2 = een/xl2
            
            # 计算梯度矩阵和向量
            plam = p0 + np.dot(P.T, lam)
            qlam = q0 + np.dot(Q.T, lam)
            gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
            GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - \
                 (diags(xlinv2.flatten(), 0).dot(Q.T)).T
            
            # 计算残差
            dpsidx = plam/ux2 - qlam/xl2
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - np.dot(a.T, lam) - epsi/z
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
                blam = dellam + dely/diagy - np.dot(GG, (delx/diagx))
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.asarray(diags(diaglamyi.flatten(), 0) + \
                    (diags(diagxinv.flatten(), 0).dot(GG.T).T).dot(GG.T))
                AAr1 = np.concatenate((Alam, a), axis=1)
                AAr2 = np.concatenate((a, -zet/z), axis=0).T
                AA = np.concatenate((AAr1, AAr2), axis=0)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m+1]
                dx = -delx/diagx - np.dot(GG.T, dlam)/diagx
            else:
                diaglamyiinv = eem/diaglamyi
                dellamyi = dellam + dely/diagy
                Axx = np.asarray(diags(diagx.flatten(), 0) + \
                    (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG))
                azz = zet/z + np.dot(a.T, (a/diaglamyi))
                axz = np.dot(-GG.T, (a/diaglamyi))
                bx = delx + np.dot(GG.T, (dellamyi/diaglamyi))
                bz = delz - np.dot(a.T, (dellamyi/diaglamyi))
                AAr1 = np.concatenate((Axx, axz), axis=1)
                AAr2 = np.concatenate((axz.T, azz), axis=1)
                AA = np.concatenate((AAr1, AAr2), axis=0)
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n+1]
                dlam = np.dot(GG, dx)/diaglamyi - dz*(a/diaglamyi) + \
                       dellamyi/diaglamyi
                
            # 计算其他变量的更新
            dy = -dely/diagy + dlam/diagy
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet*dz/z
            ds = -s + epsvecm/lam - (s*dlam)/lam
            
            # 计算步长
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)
            
            stepxx = -1.01*dxx/xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01*dx/(x-alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01*dx/(beta-x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0/stminv
            
            # 保存旧值
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            
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
                
                # 重新计算残差
                ux1 = upp-x
                xl1 = x-low
                ux2 = ux1*ux1
                xl2 = xl1*xl1
                uxinv1 = een/ux1
                xlinv1 = een/xl1
                plam = p0 + np.dot(P.T, lam)
                qlam = q0 + np.dot(Q.T, lam)
                gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
                dpsidx = plam/ux2 - qlam/xl2
                rex = dpsidx - xsi + eta
                rey = c + d*y - mu - lam
                rez = a0 - zet - np.dot(a.T, lam)
                relam = gvec - a*z - y + s - b
                rexsi = xsi*(x-alfa) - epsvecn
                reeta = eta*(beta-x) - epsvecn
                remu = mu*y - epsvecm
                rezet = zet*z - epsi
                res = lam*s - epsvecm
                
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = np.sqrt(np.dot(residu.T, residu))
                steg = steg/2
            
            residunorm = resinew.copy()
            residumax = np.max(np.abs(residu))
            steg = 2*steg
            
        epsi = 0.1*epsi
    
    # 返回最优解
    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu