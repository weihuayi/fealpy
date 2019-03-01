import numpy as np



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
