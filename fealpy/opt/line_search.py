from typing import Tuple, Optional, Callable, Union,Dict

import numpy as np
from numpy.typing import NDArray


ObjFunc = Callable[
    [NDArray],
    Tuple[np.floating, NDArray[np.floating]]
]

Float = Union[float, np.floating]

def show_line_fun(axes, fun, a, b, n=100):
    """
    @brief 绘制函数曲线
    @param axes matplotlib axes对象
    @param fun 函数对象
    @param a 起始点
    @param b 终止点
    @param n 样条数
    """
    t = np.linspace(a, b, n)
    F = np.zeros(t.shape)
    for i, ti in enumerate(t):
        f= fun(ti)
        F[i] = f
    axes.plot(t, F)

def quadratic_search(x0: NDArray, f: np.float64, d: NDArray, fun: ObjFunc,
                alpha: Optional[float]=None,**kwargs) -> Tuple[Optional[float], NDArray, np.floating, NDArray]:
    """
    @brief 二次搜索算法
    @param x0 初始点
    @param f 初始点函数值
    @param d 搜索方向
    @param fun 目标函数
    @param alpha 初始步长
    @return 最优步长alpha, 最优点x, 最优点函数值f, 最优点梯度g
    """
    a0 = 0.0

    if alpha is None:
        a1 = 2.0
    else:
        a1 = alpha

    f0 = f

    x = x0 + a1*d
    f, g = fun(x)
    f1 = f
    g1: np.float_ = np.sum(g*d)

    k = 0
    while k < 100:
        k += 1
        if np.abs(a1 - a0) > 1e-5 and a1 > 0:
            t = g1 - (f1-f0)/(a1-a0)
            a2 = a1 - 0.5*(a1-a0)*g1/t
            x = x0 + a2*d
            f, g = fun(x)
            a0 = a1
            f0 = f1

            f1 = f
            g1 = np.sum(g*d)
            a1 = float(a2)
        else:
            alpha = a1
            return alpha, x, f, g
    return alpha, x, f, g

def golden_section_search(fun, a, b, tol=1e-5):
    '''
    @brief 黄金分割搜索算法
    @param fun 函数对象, 这里的函数是关于步长alpha的函数
    @param a 区间下界
    @param b 区间上界
    @param tol 精度要求

    @return  最优步长
    '''
    phi0 = (np.sqrt(5) - 1)/2
    phi1 = (3 - np.sqrt(5))/2
    a, b = min(a, b), max(a, b)
    h = b - a
    if h <= tol:
        return (a+b)/2
    n = int(np.ceil(np.log(tol/h)/np.log(phi0)))
    c = a + phi1*h
    d = a + phi0*h
    yc = fun(c)
    yd = fun(d)

    for k in range(n):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = phi0*h
            c = a + phi1*h
            yc = fun(c)
            print("yc:%12.11g"%(yc))
        else:
            a = c
            c = d
            yc = yd
            h = phi0*h
            d = a + phi0*h
            yd = fun(d)
            print("yd:%12.11g"%(yd))
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2

def zoom(x: NDArray, s: Float, d: NDArray,
            fun: ObjFunc, alpha_0: Float, alpha_1: Float, f0: Float,
            fl: Float, c1: float, c2: float) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief 在区间内找到满足Wolfe条件的步长
    @param x 当前点 
    @param s 方向导数
    @param d 搜索方向
    @param fun 目标函数  
    @param alpha_0 区间下界
    @param alpha_1 区间上界
    @param f0 初始函数值
    @param fl 区间上界点函数值
    @param c1, c2 zoom算法参数
    @return 最优步长alpha, 最优点xc, 最优点函数值fc, 最优点梯度gc
    """
    iter_ = 0
    while iter_ < 20:
        alpha = (alpha_0 + alpha_1)/2
        xc = x + alpha*d
        fc, gc = fun(xc)
        if (fc > f0 + c1*alpha*s)\
        or (fc >= fl):
            alpha_1 = alpha
        else:
            sc = np.sum(gc*d)
            if np.abs(sc) <= -c2*s:
                return alpha, xc, fc, gc

            if sc*(alpha_1 - alpha_0) >= 0:
                alpha_1 = alpha_0
                fl = fc
            alpha_0 = alpha

        iter_ += 1
    return alpha, xc, fc, gc

def wolfe_line_search(x0: NDArray, f: Float, s: Float,
                      d: NDArray, fun: ObjFunc,
                      alpha0: Float,**kwargs) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief 强Wolfe线搜索
    @param x 当前点
    @param f 当前点函数值
    @param s 方向导数
    @param d 搜索方向
    @param fun 目标函数
    @param alpha0 初始步长
    @return: 最优步长alpha, 最优点xc, 最优点函数值fc, 最优点梯度gc
    """
    c1, c2 = 0.001, 0.1
    alpha = alpha0
    alpha_0: Float = 0.0
    alpha_1 = alpha

    fx = f
    f0 = f
    iter_ = 0

    while iter_ < 10:
        xc = x0 + alpha_1*d
        fc, gc = fun(xc)
        sc = np.sum(gc*d)

        if (fc > f0 + c1*alpha_1*s)\
        or (
            (iter_ > 0) and (fc >= fx)
        ):
            alpha, xc, fc, gc = zoom(
                x0, s, d, fun, alpha_0, alpha_1, f0, fc, c1, c2
            )
            break

        if np.abs(sc) <= -c2*s:
            alpha = alpha_1
            break

        if (sc >= 0):
            alpha, xc, fc, gc = zoom(
                x0, s, d, fun, alpha_1, alpha_0, f0, fc, c1, c2
            )
            break

        alpha_0 = alpha_1
        alpha_1 = min(10, 3*alpha_1)
        fx = fc
        iter_ = iter_ + 1
    return alpha, xc, fc, gc

def get_linesearch(key:str) -> Callable:
    if key in _linesearch_map_:
        return _linesearch_map_[key]
    else:
        raise KeyError(f"Can not find a ploter class that key '{key}' mapping to. ")

_linesearch_map_: Dict[str,Callable]={
        'wolfe':wolfe_line_search,
        'quadratic':quadratic_search}

