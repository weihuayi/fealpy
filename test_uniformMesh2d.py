array([[ 2.,  1., -1., -2.],
       [ 1.,  2., -2., -1.],
       [-1., -2.,  2.,  1.],
       [-2., -1.,  1.,  2.]])

def test_stiff_matrix()
    hx, hy = 1.123414, 3.340984
    from sympy import symbols, diff, integrate
    x, y = symbols('x, y')
    x0, y0 = x/hx, y/hy
    f0 = (1-x0)*(1-y0)
    f1 = x0*(1-y0)
    f2 = x0*y0
    f3 = (1-x0)*y0
    f = [f0, f1, f2, f3]
    nf = []
    for i in range(4):
        fx = diff(f[i], x)
        fy = diff(f[i], y)
        nf.append([fx, fy])
    M = np.zeros([4, 4], dtype=np.float_)
    S0 = np.zeros([4, 4], dtype=np.float_)
    S1 = np.zeros([4, 4], dtype=np.float_)
    for i in range(4):
        for j in range(4):
            M[i, j] = integrate(f[i]*f[j], (x, 0, hx), (y, 0, hy))
            S0[i, j] = integrate(nf[i][0]*nf[j][0], (x, 0, hx), (y, 0, hy))
            S1[i, j] = integrate(nf[i][1]*nf[j][1], (x, 0, hx), (y, 0, hy))
    print(M*36)
    print(S0*6*hx/hy)
    print(S1*6*hy/hx)

    print(f)
    print(nf)



