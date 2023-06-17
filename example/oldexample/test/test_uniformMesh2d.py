import numpy as np

a = np.array([[ 2.,  1., -1., -2.],
              [ 1.,  2., -2., -1.],
              [-1., -2.,  2.,  1.],
              [-2., -1.,  1.,  2.]])

def test_stiff_matrix():
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

def test_jump_y():
    hx, hy = 1, 1
    from sympy import symbols, diff, integrate
    x, y = symbols('x, y')
    x0, y0 = x/hx, y/hy

    f0 = (1-x0)*(1-y0)
    f1 = (1-x0)*y0
    f2 = x0*(1-y0)
    f3 = x0*y0
    F0 = [f0, f1, f2, f3]
    F1 = [F0[i].subs({x:x+hx}) for i in range(4)]

    f_l = F1 + [0*x, 0*x]
    f_r = [0*x, 0*x] + F0

    nf_r = []
    nnf_r = []
    nf_l = []
    nnf_l = []
    for i in range(6):
        f_lx = diff(f_l[i], x)
        f_ly = diff(f_l[i], y)
        f_lxx = diff(f_lx, x)
        f_lxy = diff(f_lx, y)
        f_lyy = diff(f_ly, y)
        nf_l.append([f_lx.subs({x:0}), f_ly.subs({x:0})])
        nnf_l.append([f_lxx.subs({x:0}), f_lxy.subs({x:0}), f_lyy.subs({x:0})])

        f_rx = diff(f_r[i], x)
        f_ry = diff(f_r[i], y)
        f_rxx = diff(f_rx, x)
        f_rxy = diff(f_rx, y)
        f_ryy = diff(f_ry, y)
        nf_r.append([f_rx.subs({x:0}), f_ry.subs({x:0})])
        nnf_r.append([f_rxx.subs({x:0}), f_rxy.subs({x:0}), f_ryy.subs({x:0})])

    print(nf_l)
    print(nnf_l)
    print(nf_r)
    print(nnf_r)

    N0 = np.zeros([6, 6], dtype=np.float_)
    N1 = np.zeros([6, 6], dtype=np.float_)
    N00 = np.zeros([6, 6], dtype=np.float_)
    N10 = np.zeros([6, 6], dtype=np.float_)
    N11 = np.zeros([6, 6], dtype=np.float_)
    for i in range(6):
        for j in range(6):
            g0 = (nf_l[i][0]-nf_r[i][0])*(nf_l[j][0]-nf_r[j][0])
            g1 = (nf_l[i][1]-nf_r[i][1])*(nf_l[j][1]-nf_r[j][1])

            g00 = (nnf_l[i][0]-nnf_r[i][0])*(nnf_l[j][0]-nnf_r[j][0])
            g10 = (nnf_l[i][1]-nnf_r[i][1])*(nnf_l[j][1]-nnf_r[j][1])
            g11 = (nnf_l[i][2]-nnf_r[i][2])*(nnf_l[j][2]-nnf_r[j][2])
            N0[i, j] = integrate(g0, (y, 0, hy))
            N1[i, j] = integrate(g1, (y, 0, hy))
            N00[i, j] = integrate(g00, (y, 0, hy))
            N10[i, j] = integrate(g10, (y, 0, hy))
            N11[i, j] = integrate(g11, (y, 0, hy))
    print(N0*6*(hx**2)/hy)
    #print(N1*hy)
    print(N10*hx**2*hy)

def test_jump_x():
    hx, hy = 1, 1
    from sympy import symbols, diff, integrate
    x, y = symbols('x, y')
    x0, y0 = x/hx, y/hy

    f0 = (1-x0)*(1-y0)
    f1 = (1-x0)*y0
    f2 = x0*(1-y0)
    f3 = x0*y0
    F0 = [f0, f1, f2, f3]
    F1 = [F0[i].subs({y:y+hy}) for i in range(4)]

    f_d = [F1[0], F1[2], F1[1], F1[3], 0*y, 0*y]
    f_u = [0*y, 0*y, F0[0], F0[2], F0[1], F0[3]]

    nf_u = []
    nnf_u = []
    nf_d = []
    nnf_d = []
    for i in range(6):
        f_dx = diff(f_d[i], x)
        f_dy = diff(f_d[i], y)
        f_dxx = diff(f_dx, x)
        f_dxy = diff(f_dx, y)
        f_dyy = diff(f_dy, y)
        nf_d.append([f_dx.subs({y:0}), f_dy.subs({y:0})])
        nnf_d.append([f_dxx.subs({y:0}), f_dxy.subs({y:0}), f_dyy.subs({y:0})])

        f_ux = diff(f_u[i], x)
        f_uy = diff(f_u[i], y)
        f_uxx = diff(f_ux, x)
        f_uxy = diff(f_ux, y)
        f_uyy = diff(f_uy, y)
        nf_u.append([f_ux.subs({y:0}), f_uy.subs({y:0})])
        nnf_u.append([f_uxx.subs({y:0}), f_uxy.subs({y:0}), f_uyy.subs({y:0})])

    print(nf_d)
    print(nf_u)
    print(nnf_d)
    print(nnf_u)

    N0 = np.zeros([6, 6], dtype=np.float_)
    N1 = np.zeros([6, 6], dtype=np.float_)
    N00 = np.zeros([6, 6], dtype=np.float_)
    N10 = np.zeros([6, 6], dtype=np.float_)
    N11 = np.zeros([6, 6], dtype=np.float_)
    for i in range(6):
        for j in range(6):
            g0 = (nf_d[i][0]-nf_u[i][0])*(nf_d[j][0]-nf_u[j][0])
            g1 = (nf_d[i][1]-nf_u[i][1])*(nf_d[j][1]-nf_u[j][1])

            g00 = (nnf_d[i][0]-nnf_u[i][0])*(nnf_d[j][0]-nnf_u[j][0])
            g10 = (nnf_d[i][1]-nnf_u[i][1])*(nnf_d[j][1]-nnf_u[j][1])
            g11 = (nnf_d[i][2]-nnf_u[i][2])*(nnf_d[j][2]-nnf_u[j][2])
            N0[i, j] = integrate(g0, (x, 0, hx))
            N1[i, j] = integrate(g1, (x, 0, hx))
            N00[i, j] = integrate(g00, (x, 0, hx))
            N10[i, j] = integrate(g10, (x, 0, hx))
            N11[i, j] = integrate(g11, (x, 0, hx))

    #print(N0*hy)
    print(N1*6*hy**2/hx)
    print(N10*hy**2*hx)

test_jump_y()
test_jump_x()

