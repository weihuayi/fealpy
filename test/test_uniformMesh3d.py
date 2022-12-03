import numpy as np
import sys

def test_stiff_matrix(hx, hy, hz):
    from sympy import symbols, diff, integrate
    x, y, z = symbols('x, y, z')
    x0, y0, z0 = x/hx, y/hy, z/hz
    f0 = (1-x0)*(1-y0)*(1-z0)
    f1 = (1-x0)*y0*(1-z0)
    f2 = (1-x0)*y0*z0
    f3 = (1-x0)*(1-y0)*z0
    f4 = x0*(1-y0)*(1-z0)
    f5 = x0*y0*(1-z0)
    f6 = x0*y0*z0
    f7 = x0*(1-y0)*z0
    f = [f0, f1, f2, f3, f4, f5, f6, f7]
    nf = []
    for i in range(8):
        fx = diff(f[i], x)
        fy = diff(f[i], y)
        fz = diff(f[i], z)
        nf.append([fx, fy, fz])
    M = np.zeros([8, 8], dtype=np.float_)
    S0 = np.zeros([8, 8], dtype=np.float_)
    S1 = np.zeros([8, 8], dtype=np.float_)
    S2 = np.zeros([8, 8], dtype=np.float_)
    for i in range(8):
        for j in range(8):
            M[i, j] = integrate(f[i]*f[j], (x, 0, hx), (y, 0, hy), (z, 0, hz))
            S0[i, j] = integrate(nf[i][0]*nf[j][0], (x, 0, hx), (y, 0, hy), (z,
                0, hz))
            S1[i, j] = integrate(nf[i][1]*nf[j][1], (x, 0, hx), (y, 0, hy),
                    (z,0,hz))
            S2[i, j] = integrate(nf[i][2]*nf[j][2], (x, 0, hx), (y, 0,
                hy),(z,0,hz))
    print(M*36*6/(hx*hy*hz))
    print(S0*36*hx/(hy*hz))
    print(S1*36*hy/(hx*hz))
    print(S2*36*hz/(hx*hy))

def test_nabla2_matrix(hx, hy, hz):
    from sympy import symbols, diff, integrate
    x, y, z = symbols('x, y, z')
    x0, y0, z0 = x/hx, y/hy, z/hz
    f0 = (1-x0)*(1-y0)*(1-z0)
    f1 = (1-x0)*y0*(1-z0)
    f2 = (1-x0)*y0*z0
    f3 = (1-x0)*(1-y0)*z0
    f4 = x0*(1-y0)*(1-z0)
    f5 = x0*y0*(1-z0)
    f6 = x0*y0*z0
    f7 = x0*(1-y0)*z0
    f = [f0, f1, f2, f3, f4, f5, f6, f7]
    nnf = []
    for i in range(8):
        fx = diff(f[i], x)
        fy = diff(f[i], y)
        fz = diff(f[i], z)

        fxx = diff(fx, x)
        fxy = diff(fx, y)
        fxz = diff(fx, z)
        fyy = diff(fy, y)
        fyz = diff(fy, z)
        fzz = diff(fz, z)
        nnf.append([fxx, fxy, fxz, fyy, fyz, fzz])

    S00 = np.zeros([8, 8], dtype=np.float_)
    S01 = np.zeros([8, 8], dtype=np.float_)
    S02 = np.zeros([8, 8], dtype=np.float_)
    S11 = np.zeros([8, 8], dtype=np.float_)
    S12 = np.zeros([8, 8], dtype=np.float_)
    S22 = np.zeros([8, 8], dtype=np.float_)
    for i in range(8):
        for j in range(8):
            S00[i, j] = integrate(nnf[i][0]*nnf[j][0], (x, 0, hx), (y, 0, hy), (z,
                0, hz))
            S01[i, j] = integrate(nnf[i][1]*nnf[j][1], (x, 0, hx), (y, 0, hy),
                    (z,0,hz))
            S02[i, j] = integrate(nnf[i][2]*nnf[j][2], (x, 0, hx), (y, 0,
                hy),(z,0,hz))
            S11[i, j] = integrate(nnf[i][3]*nnf[j][3], (x, 0, hx), (y, 0,
                hy),(z,0,hz))
            S12[i, j] = integrate(nnf[i][4]*nnf[j][4], (x, 0, hx), (y, 0,
                hy),(z,0,hz))
            S22[i, j] = integrate(nnf[i][5]*nnf[j][5], (x, 0, hx), (y, 0,
                hy),(z,0,hz))
    print(S00*6*hx**3/(hy*hz))
    print(S01*6*hx*hy/hz)
    print(S02*6*hx*hz/hy)
    print(S11*6*hy**3/(hx*hz))
    print(S12*6*hy*hz/hx)
    print(S22*6*hz**3/(hx*hy))

    #print(f)
    #print(nf)

def test_poisson():

    box = [0, 1, 0, 1, 0, 1]
    N = int(sys.argv[1])
    h = [1/N, 1/N, 1/N]
    origin = [0, 0, 0]
    extend = [0, N, 0, N, 0, N]

    meshb = UniformMesh3d(extend, h, origin) # 背景网格
    pnode = meshb.entity('node').reshape(-1, 3)

    J = meshb.grad_jump_matrix()
    S = meshb.stiff_matrix()
    M = meshb.mass_matrix()
    G2 = meshb.grad_2_matrix()

    F = meshb.source_vector(source)
    x = meshb.function().reshape(-1)

    isDDof = meshb.boundary_node_flag()
    x[isDDof] = exu(pnode[isDDof])

    F -= S@x
    bdIdx = np.zeros(S.shape[0], dtype=np.int_)
    bdIdx[isDDof] = 1
    Tbd = spdiags(bdIdx, 0, S.shape[0], S.shape[0])
    T = spdiags(1-bdIdx, 0, S.shape[0], S.shape[0])
    S = T@S@T + Tbd
    F[isDDof] = x[isDDof]
    print(np.sum(isDDof))

    x = spsolve(S, F)
    print(np.max(np.abs(x - exu(pnode))))

def test_jump_x(hx, hy, hz):
    from sympy import symbols, diff, integrate
    x, y, z = symbols('x, y, z')
    x0, y0, z0 = x/hx, y/hy, z/hz
    
    f0 = (1-x0)*(1-y0)*(1-z0)
    f1 = (1-x0)*(1-y0)*z0
    f2 = (1-x0)*y0*(1-z0)
    f3 = (1-x0)*y0*z0
    f4 = x0*(1-y0)*(1-z0)
    f5 = x0*(1-y0)*z0
    f6 = x0*y0*(1-z0)
    f7 = x0*y0*z0
    F0 = [f0, f1, f2, f3, f4, f5, f6, f7]
    nf = []

    F1 = [F0[i].subs({x:x+hx}) for i in range(8)] # 将 x 替换成 x+hx

    f_l = F1 + [0*x, 0*x, 0*x, 0*x]
    f_r = [0*x, 0*x, 0*x, 0*x] + F0

    nf_r = []
    nf_l = []
    for i in range(12):
        f_lx = diff(f_l[i], x)
        f_ly = diff(f_l[i], y)
        f_lz = diff(f_l[i], z)
        nf_l.append([f_lx.subs({x:0}), f_ly.subs({x:0}), f_lz.subs({x:0})])

        f_rx = diff(f_r[i], x)
        f_ry = diff(f_r[i], y)
        f_rz = diff(f_r[i], z)
        nf_r.append([f_rx.subs({x:0}), f_ry.subs({x:0}), f_rz.subs({x:0})])

    '''
    print(nf_l)
    print(nnf_l)
    print(nf_r)
    print(nnf_r)
    '''

    N0 = np.zeros([12, 12], dtype=np.float_)
    N1 = np.zeros([12, 12], dtype=np.float_)
    N2 = np.zeros([12, 12], dtype=np.float_)

    for i in range(12):
        for j in range(12):
            g0 = (nf_l[i][0]-nf_r[i][0])*(nf_l[j][0]-nf_r[j][0])
            g1 = (nf_l[i][1]-nf_r[i][1])*(nf_l[j][1]-nf_r[j][1])
            g2 = (nf_l[i][2]-nf_r[i][2])*(nf_l[j][2]-nf_r[j][2])

            N0[i, j] = integrate(g0, (y, 0, hy),(z, 0, hz))
            N1[i, j] = integrate(g1, (y, 0, hy),(z, 0, hz))
            N2[i, j] = integrate(g2, (y, 0, hy),(z, 0, hz))

    #print(N0*6*(hx**2)/hy)
    #print(N1*hy)
    #print(N10*hx**2*hy)
    print(N0*36*hx**2/(hy*hz))


def test_jump_y(hx, hy, hz):
    from sympy import symbols, diff, integrate
    x, y, z = symbols('x, y, z')
    x0, y0, z0 = x/hx, y/hy, z/hz
    
    f0 = (1-x0)*(1-y0)*(1-z0)
    f1 = (1-x0)*(1-y0)*z0
    f2 = (1-x0)*y0*(1-z0)
    f3 = (1-x0)*y0*z0
    f4 = x0*(1-y0)*(1-z0)
    f5 = x0*(1-y0)*z0
    f6 = x0*y0*(1-z0)
    f7 = x0*y0*z0
    F0 = [f0, f1, f2, f3, f4, f5, f6, f7]
    nf = []

    F1 = [F0[i].subs({y:y+hy}) for i in range(8)] # 将 y 替换成 y+hy

    f_l = [F1[4], F1[5], F1[0], F1[1], F1[6], F1[7], F1[2], F1[3], 0*y, 0*y,
            0*y, 0*y]
    f_r = [0*y, 0*y, 0*y, 0*y, F0[4], F0[5], F0[0], F0[1], F0[6], F0[7], F0[2],
            F0[3]]

    nf_r = []
    nf_l = []
    for i in range(12):
        f_lx = diff(f_l[i], x)
        f_ly = diff(f_l[i], y)
        f_lz = diff(f_l[i], z)
        nf_l.append([f_lx.subs({y:0}), f_ly.subs({y:0}), f_lz.subs({y:0})])

        f_rx = diff(f_r[i], x)
        f_ry = diff(f_r[i], y)
        f_rz = diff(f_r[i], z)
        nf_r.append([f_rx.subs({y:0}), f_ry.subs({y:0}), f_rz.subs({y:0})])

    N0 = np.zeros([12, 12], dtype=np.float_)
    N1 = np.zeros([12, 12], dtype=np.float_)
    N2 = np.zeros([12, 12], dtype=np.float_)
    for i in range(12):
        for j in range(12):
            g0 = (nf_l[i][0]-nf_r[i][0])*(nf_l[j][0]-nf_r[j][0])
            g1 = (nf_l[i][1]-nf_r[i][1])*(nf_l[j][1]-nf_r[j][1])
            g2 = (nf_l[i][2]-nf_r[i][2])*(nf_l[j][2]-nf_r[j][2])

            N0[i, j] = integrate(g0, (x, 0, hx),(z, 0, hz))
            N1[i, j] = integrate(g1, (x, 0, hx),(z, 0, hz))
            N2[i, j] = integrate(g2, (x, 0, hx),(z, 0, hz))

    #print(N0*6*(hx**2)/hy)
    #print(N1*hy)
    #print(N10*hx**2*hy)
    print(N1*36*hy**2/(hz*hx))


def test_jump_z(hx, hy, hz):
    from sympy import symbols, diff, integrate
    x, y, z = symbols('x, y, z')
    x0, y0, z0 = x/hx, y/hy, z/hz
    
    f0 = (1-x0)*(1-y0)*(1-z0)
    f1 = (1-x0)*(1-y0)*z0
    f2 = (1-x0)*y0*(1-z0)
    f3 = (1-x0)*y0*z0
    f4 = x0*(1-y0)*(1-z0)
    f5 = x0*(1-y0)*z0
    f6 = x0*y0*(1-z0)
    f7 = x0*y0*z0
    F0 = [f0, f1, f2, f3, f4, f5, f6, f7]
    nf = []

    F1 = [F0[i].subs({z:z+hz}) for i in range(8)] # 将 z 替换成 z+hz

    f_l = [F1[0], F1[2], F1[4], F1[6], F1[1], F1[3], F1[5], F1[7], 0*y, 0*y,
            0*y, 0*y] 
    f_r = [0*y, 0*y, 0*y, 0*y, F0[0], F0[2], F0[4], F0[6], F0[1], F0[3], F0[5], F0[7]] 

    nf_r = []
    nf_l = []
    for i in range(12):
        f_lx = diff(f_l[i], x)
        f_ly = diff(f_l[i], y)
        f_lz = diff(f_l[i], z)
        nf_l.append([f_lx.subs({z:0}), f_ly.subs({z:0}), f_lz.subs({z:0})])

        f_rx = diff(f_r[i], x)
        f_ry = diff(f_r[i], y)
        f_rz = diff(f_r[i], z)
        nf_r.append([f_rx.subs({z:0}), f_ry.subs({z:0}), f_rz.subs({z:0})])

    '''
    print(nf_l)
    print(nnf_l)
    print(nf_r)
    print(nnf_r)
    '''

    N0 = np.zeros([12, 12], dtype=np.float_)
    N1 = np.zeros([12, 12], dtype=np.float_)
    N2 = np.zeros([12, 12], dtype=np.float_)

    for i in range(12):
        for j in range(12):
            g0 = (nf_l[i][0]-nf_r[i][0])*(nf_l[j][0]-nf_r[j][0])
            g1 = (nf_l[i][1]-nf_r[i][1])*(nf_l[j][1]-nf_r[j][1])
            g2 = (nf_l[i][2]-nf_r[i][2])*(nf_l[j][2]-nf_r[j][2])

            N0[i, j] = integrate(g0, (x, 0, hx),(y, 0, hy))
            N1[i, j] = integrate(g1, (x, 0, hx),(y, 0, hy))
            N2[i, j] = integrate(g2, (x, 0, hx),(y, 0, hy))

    #print(N0*6*(hx**2)/hy)
    #print(N1*hy)
    #print(N10*hx**2*hy)
    print(N2*36*hz**2/hx/hy)


hx = float(sys.argv[1])
hy = float(sys.argv[2])
hz = float(sys.argv[3])

test_stiff_matrix(hx, hy, hz)
#test_nabla2_matrix(hx, hy, hz)
#test_jump_x(hx, hy, hz)
#test_jump_y(hx, hy, hz)
#test_jump_z(hx, hy, hz)


