from ...backend import backend_manager as bm

def penalty(value):
    penalty_factor = 10e+6
    penalty = 0 + ((0 < value) * (value < 1)) * value + (value >= 1) * (value ** 2)
    return penalty_factor * penalty

def spring(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    f = (x3 + 2) * x2 * x1**2
    g1 = 1 - ((x2**3) * x3) / (71785 * (x1**4))
    g2 = (4 * (x2**2) - x1 * x2) / (12566 * (x2 * (x1**3) - (x1**4))) + 1 / (5108 * (x1**2)) - 1
    g3 = 1 - (140.45 * x1) / ((x2**2) * x3)
    g4 = ((x1 + x2) / 1.5) - 1
    return f + (penalty(g1) + penalty(g2) + penalty(g3) + penalty(g4))

def pvd(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]

    f = (
        0.6224 * x1 * x3 * x4 +
        1.7781 * x2 * x3**2 +
        3.1661 * x1**2 * x4 +
        19.84 * x1**2 * x3
    )

    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -bm.pi * x3**2 * x4 - (4/3) * bm.pi * x3**3 + 1296000
    g4 = x4 - 240

    penalties = penalty(g1) + penalty(g2) + penalty(g3) + penalty(g4)
    return f + penalties

def speed_reducer(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]
    x6 = x[:, 5]
    x7 = x[:, 6]

    f = (
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )

    g1 = 27 / (x1 * x2**2 * x3) - 1
    g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
    g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
    g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
    g5 = (1 / (110 * x6**3)) * bm.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) - 1
    g6 = (1 / (85 * x7**3)) * bm.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) - 1
    g7 = (x2 * x3) / 40 - 1
    g8 = (5 * x2) / x1 - 1
    g9 = x1 / (12 * x2) - 1
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    g11 = (1.1 * x7 + 1.9) / x5 - 1

    penalties = (
        penalty(g1) + penalty(g2) + penalty(g3) + penalty(g4) +
        penalty(g5) + penalty(g6) + penalty(g7) + penalty(g8) +
        penalty(g9) + penalty(g10) + penalty(g11)
    )

    return f + penalties

def heat_exchanger_case1(x, epsilon = 1e-8, inf = 1e+10):
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8]
    
    f = 35 * x1**0.6 + 35 * x2**0.6
    
    h1 = 200 * x1 * x4 - x3
    h2 = 200 * x2 * x6 - x5
    h3 = x3 - 10000 * (x7 - 100)
    h4 = x5 - 10000 * (300 - x7)
    h5 = x3 - 10000 * (600 - x8)
    h6 = x5 - 10000 * (900 - x9)
    h7 = x4 * (bm.log(bm.maximum(epsilon, (x8 - 100))) - bm.log(bm.maximum(epsilon, (600 - x7)))) - x8 + x7 + 500
    mark = x9 <= x7
    h8 = bm.where(mark, inf, x6 * (bm.log(bm.maximum(epsilon, (x9 - x7))) - bm.log(600)) - x9 + x7 + 600) 

    penalties = (
        penalty(bm.abs(h1)) + penalty(bm.abs(h2)) + penalty(bm.abs(h3)) + penalty(bm.abs(h4)) +
        penalty(bm.abs(h5)) + penalty(bm.abs(h6)) + penalty(bm.abs(h7)) + penalty(bm.abs(h8))
    )

    return f + penalties

def three_bar(x):
    l = 100.0
    P = 2.0
    q = 2.0

    x1 = x[:, 0]
    x2 = x[:, 1]

    f = l * (2 * bm.sqrt(2) * x1 + x2)

    g1 = P * (bm.sqrt(2) * x1 + x2) / (bm.sqrt(2) * x1**2 + 2 * x1 * x2 + 1e-8) - q
    g2 = P * x2 / (bm.sqrt(2) * x1**2 + 2 * x1 * x2 + 1e-8) - q
    g3 = P / (bm.sqrt(2) * x2 + x1 + 1e-8) - q

    penalties = penalty(g1) + penalty(g2) + penalty(g3)
    return f + penalties


def welded_beam(x):
    P = 6000.0
    L = 14.0
    E = 30e6
    G = 12e6
    tmax = 13600.0
    sigma_max = 30000.0
    deta_max = 0.25

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]

    # M, R, J
    M = P * (L + 0.5 * x2)
    r1 = (x2**2) / 4
    r2 = ((x1 + x3) / 2) ** 2
    R = bm.sqrt(r1 + r2)
    j1 = bm.sqrt(2) * x1 * x2
    j2 = (x2**2) / 4
    j3 = ((x1 + x3) / 2) ** 2
    J = 2 * j1 * (j2 + j3)

    # stresses
    sigma_x = 6 * P * L / (x4 * x3**2)
    deta_x = 6 * P * L**3 / (E * x4 * x3**2)

    # buckling load
    p1 = (4.013 * E * bm.sqrt((x3**2 * x4**6) / 36)) / (L**2)
    p2 = (x3 / (2 * L)) * bm.sqrt(E / (4 * G))
    Pc = p1 * (1 - p2)

    # shear
    t1 = P / bm.sqrt(2 * x1 * x2)
    t2 = M * R / J
    t = bm.sqrt(t1**2 + 2 * t1 * t2 * (x2 / (2 * R)) + t2**2)

    # objective
    f = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)

    # constraints
    g1 = t - tmax
    g2 = sigma_x - sigma_max
    g3 = deta_x - deta_max
    g4 = x1 - x4
    g5 = P - Pc

    penalties = (
        penalty(g1) + penalty(g2) + penalty(g3) +
        penalty(g4) + penalty(g5)
    )

    return f + penalties

def gear_train(x):
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    target_ratio = 1 / 6.931
    ratio = (x1 * x2) / (x3 * x4 + 1e-8)  # 避免除以0
    return (target_ratio - ratio) ** 2

def rolling_element_bearing(x):
    D, d, Bw = 160, 90, 30

    Dm = x[:, 0]
    Db = x[:, 1]
    Z = bm.round(x[:, 2])
    fi = x[:, 3]
    f0 = x[:, 4]
    KDmin = x[:, 5]
    KDmax = x[:, 6]
    ep = x[:, 7]
    ee = x[:, 8]
    xi = x[:, 9]

    T = D - d - 2 * Db

    part1 = (((D - d) / 2) - 3 * (T / 4)) ** 2
    part2 = (D / 2 - T / 4 - Db) ** 2
    part3 = (d / 2 + T / 4) ** 2
    denom = 2 * ((D - d) / 2 - 3 * (T / 4)) * (D / 2 - T / 4 - Db)

    # phio 限定在 [-1, 1] 避免 acos 数值错误
    arg = bm.clip((part1 + part2 - part3) / denom, -1.0, 1.0)
    phio = 2 * bm.pi - bm.acos(arg)

    g1 = Z - 1 - phio / (2 * bm.arcsin(Db / Dm))
    g2 = -2 * Db + KDmin * (D - d)
    g3 = -KDmax * (D - d) + 2 * Db
    g4 = -xi * Bw + Db
    g5 = -Dm + 0.5 * (D + d)
    g6 = -(0.5 + ee) * (D + d) + Dm
    g7 = -0.5 * (D - Dm - Db) + ep * Db

    penalties = (
        penalty(g1) + penalty(g2) + penalty(g3) + 
        penalty(g4) + penalty(g5) + penalty(g6) + penalty(g7) 
    )

    gama = Db / Dm
    factor = ((1 + 1.04 * ((1 - gama) / (1 + gama)) ** 1.72 *
               ((fi * (2 * f0 - 1) / f0 / (2 * fi - 1)) ** 0.41)) ** (10 / 3)) ** -0.3
    term1 = 37.91 * factor
    term2 = (gama ** 0.3 * (1 - gama) ** 1.39) / (1 + gama) ** (1 / 3)
    term3 = (2 * fi / (2 * fi - 1)) ** 0.41
    fc = term1 * term2 * term3

    condition = Db <= 25.4
    f = bm.where(condition,
                 fc * Z ** (2 / 3) * Db ** 1.8,
                 3.647 * fc * Z ** (2 / 3) * Db ** 1.4)

    return -f + penalties

def multiple_disk(x):
    x = bm.round(x)
    x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    Mf = 3
    Ms = 40
    Iz = 55
    n = 250
    Tmax = 15
    s = 1.5
    delta = 0.5
    Vsrmax = 10
    rho = 0.0000078
    pmax = 1
    mu = 0.6
    Lmax = 30
    delR = 20

    r2 = x2
    r1 = x1
    Mh = 2 / 3 * mu * x4 * x5 * (r2**3 - r1**3) / (r2**2 - r1**2 + 1e-8)
    Prz = x4 / (bm.pi * (r2**2 - r1**2 + 1e-8))
    Vsr = (2 * bm.pi * n / 90) * (r2**3 - r1**3) / (r2**2 - r1**2 + 1e-8)
    T = (Iz * bm.pi * n / 30) / (Mh + Mf + 1e-8)

    g1 = -x2 + x1 + delR
    g2 = (x5 + 1) * (x3 + delta) - Lmax
    g3 = Prz - pmax
    g4 = Prz * Vsr - pmax * Vsrmax
    g5 = Vsr - Vsrmax
    g6 = T - Tmax
    g7 = s * Ms - Mh
    g8 = -T
    
    f = bm.pi * (r2**2 - r1**2) * x3 * (x5 + 1) * rho

    penalties = (penalty(g1) + penalty(g2) + penalty(g3) + penalty(g4) +
                penalty(g5) + penalty(g6) + penalty(g7) + penalty(g8)              
    )
    return f + penalties

constrained_benchmark_data = [
    {
        "objective": multiple_disk,
        "lb": bm.array([60, 90, 1, 0, 2]),
        "ub": bm.array([80, 110, 3, 1000, 9]),
        "ndim": 5,
        "optimal": 2.3524245790E-01
    },
    {
        "objective": rolling_element_bearing,
        "lb": bm.array([
            0.5 * (160 + 90),   
            0.15 * (160 - 90),  
            4,                  
            0.515,              
            0.515,              
            0.4,                
            0.6,                
            0.3,                
            0.02,               
            0.6                 
        ]),
        "ub": bm.array([
            0.6 * (160 + 90),   
            0.45 * (160 - 90),  
            50,                 
            0.6,                
            0.6,                
            0.5,                
            0.7,                
            0.4,                
            0.1,                
            0.85                
        ]),
        "ndim": 10,
        "optimal": 1.4614135715E+04
    },
    {
        "objective": gear_train,
        "lb": bm.array([12, 12, 12, 12]),
        "ub": bm.array([60, 60, 60, 60]),
        "ndim": 4,
        "optimal": 0.0000000000E+00
    },
    {
        "objective": welded_beam,
        "lb": bm.array([0.125, 0.1, 0.1, 0.1]),
        "ub": bm.array([2, 10, 10, 2]),
        "ndim": 4,
        "optimal": 1.6702177263E+00
    },
    {
        "objective": three_bar,
        "lb": bm.array([0, 0]),
        "ub": bm.array([1, 1]),
        "ndim": 2,
        "optimal": 2.6389584338E+02 
    },
    {
        "objective": heat_exchanger_case1,
        "lb": bm.array([0,     0,     0,     0,      1000, 0,   100, 100, 100]),
        "ub": bm.array([10,  200,   100,   200,  2_000_000, 600, 600, 600, 900]),
        "ndim": 9,
        "optimal": 1.8931162966E+02
    },
    {
        "objective": spring,
        "lb": bm.array([0.05, 0.25, 2]),
        "ub": bm.array([2, 1.3, 15]),
        "ndim": 3,
        "optimal": 1.2665232788E-02
    },
    {
        "objective": pvd,
        "lb": bm.array([0, 0, 10, 10]),
        "ub": bm.array([99, 99, 200, 200]),
        "ndim": 4,
        "optimal": 5.8853327736E+03  
    },
    {
        "objective": speed_reducer,
        "lb": bm.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]),
        "ub": bm.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]),
        "ndim": 7,
        "optimal": 2.9944244658E+03
    },
]