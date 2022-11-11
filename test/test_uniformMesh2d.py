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

def test_poisson():

    box = [0, 1, 0, 1]
    N = int(sys.argv[1])
    h = [1/N, 1/N]
    origin = [-1/4/N, -1/4/N]
    origin = [0, 0]
    extend = [0, N+1, 0, N+1]

    meshb = UniformMesh2d(extend, h, origin) # 背景网格
    mesht = MeshFactory.triangle([0, 1, 0, 1], 1)
    tnode = mesht.entity('node')
    tval = ff(tnode)

    t2b(mesht, meshb, tval)
    meshb.stiff_matrix()
    pnode = meshb.entity('node').reshape(-1, 2)

    F = meshb.source_vector(source)
    A = meshb.stiff_matrix()
    x = meshb.function().reshape(-1)

    isDDof = meshb.ds.boundary_node_flag()
    x[isDDof] = exu(pnode[isDDof])

    F -= A@x
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isDDof] = 1
    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    A = T@A@T + Tbd
    F[isDDof] = x[isDDof]

    x = spsolve(A, F)
    print(np.max(np.abs(x - exu(pnode))))



