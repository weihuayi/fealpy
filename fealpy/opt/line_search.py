from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from .optimizer_base import ObjFunc, Float


def show_line_fun(axes, fun, a, b, n=100):
    t = np.linspace(a, b, n)
    F = np.zeros(t.shape)
    for i, ti in enumerate(t):
        f= fun(ti)
        F[i] = f
    axes.plot(t, F)

def quadratic_search(fun, f0, g0, alpha=2):
    t0 = 0
    t1 = alpha

    minf = f0
    alpha = 0

    NF = 0
    [f1, g1] = fun(t1)
    print("t1:", t1, "f1:", f1)
    NF += 1

    if minf > f1:
        alpha = t1
        minf = f1
        print("alpha:", alpha, "f:", minf)

    k = 0
    while k < 2:
        k += 1
        t = g1 - (f1 - f0)/(t1 - t0)
        t2 = t1 - 0.5*(t1 - t0)*g1/t

        if t2 < 0:
            break

        [f, g] = fun(t2)
        print("t2:", t2, "f2:", f)

        NF += 1
        if minf > f:
            alpha = t2
            minf = f
            print("alpha:", alpha, "f:", minf)

        t0 = t1
        f0 = f1
        g0 = g1

        t1 = t2
        f1 = f
        g1 = g

    return alpha, NF

def golden_section_search(fun, a, b, tol=1e-5):
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


def line_search(x0: NDArray, f: np.float_, d: NDArray, fun: ObjFunc,
                alpha: Optional[float]=None) -> Tuple[Optional[float], NDArray, np.floating, NDArray]:
    """
    @brief

    @return: alpha, x, f, g
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


def zoom(x: NDArray, s: Float, d: NDArray,
            fun: ObjFunc, alpha_0: Float, alpha_1: Float, f0: Float,
            fl: Float, c1: float, c2: float) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief

    @return alpha, xc, fc, gc
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
                alpha_0, alpha_1 = alpha, alpha_0
                fl = fc

        iter_ += 1
    return alpha, xc, fc, gc


def wolfe_line_search(x: NDArray, f: Float, s: Float,
                      d: NDArray, fun: ObjFunc,
                      alpha0: Float) -> Tuple[Float, NDArray, np.floating, NDArray]:
    """
    @brief

    @return: alpha, xc, fc, gc
    """
    c1, c2 = 0.001, 0.1
    alpha = alpha0
    alpha_0: Float = 0.0
    alpha_1 = alpha

    fx = f
    f0 = f
    iter_ = 0

    while iter_ < 10:
        xc = x + alpha_1*d
        fc, gc = fun(xc)
        sc = np.sum(gc*d)

        if (fc > f0 + c1*alpha_1*s)\
        or (
            (iter_ > 0) and (fc >= fx)
        ):
            alpha, xc, fc, gc = zoom(
                x, s, d, fun, alpha_0, alpha_1, f0, fc, c1, c2
            )
            break

        if np.abs(sc) <= -c2*s:
            alpha = alpha_1
            break

        if (sc >= 0):
            alpha, xc, fc, gc = zoom(
                x, s, d, fun, alpha_1, alpha_0, f0, fc, c1, c2
            )
            break

        alpha_0 = alpha_1
        alpha_1 = min(10, 3*alpha_1)
        fx = fc
        iter_ = iter_ + 1
    return alpha, xc, fc, gc
