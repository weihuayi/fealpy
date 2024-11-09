from fealpy.backend import backend_manager as bm

def F1(x):
    """
    Step
    """
    return bm.sum((x + 0.5) ** 2, axis=-1)

def F2(x):
    """
    Sphere
    """
    return bm.sum(x ** 2, axis=-1)

def F3(x):
    """
    Sum Quares
    """
    dim = x.shape[-1]
    return bm.sum(bm.arange(1, dim + 1) * (x ** 2), axis=-1)

def F4(x):
    """
    Bent Cigar
    """
    return x[:,0] ** 2 + bm.sum(1e6 * x[:,1:] ** 2, axis=-1)

def F5(x):
    """
    Beale
    """
    return (1.5 - x[:,0] + x[:,0] * x[:,1]) ** 2 + (2.25 - x[:,0] + x[:,0] * x[:,1] ** 2) ** 2 + (2.625 - x[:,0] + x[:,0] * x[:,1] ** 3) ** 2

def F6(x):
    """
    Eason
    """
    return -bm.cos(x[:,0]) * bm.cos(x[:,1]) * bm.exp(-((x[:,0] - bm.pi) ** 2 + (x[:,1] - bm.pi) ** 2))

def F7(x):
    """
    Matyas
    """
    return 0.26 * (x[:,0] ** 2 + x[:,1] ** 2) - 0.48 * x[:,0] * x[:,1]

def F8(x):
    """
    Colville
    """
    return 100 * (x[:,0] ** 2 - x[:,1]) ** 2 + (x[:,0] - 1) ** 2 + (x[:,2] - 1) ** 2 + 90 * (x[:,2] ** 2 - x[:,3]) ** 2 + 10.1 * ((x[:,1] - 1) ** 2 + (x[:,3] - 1) ** 2) + 19.8 * (x[:,1] - 1) * (x[:,3] - 1)

def F9(x):
    """
    Zakharov
    """
    dim = x.shape[-1]
    return bm.sum(x ** 2, axis= -1) + (bm.sum(0.5 * bm.arange(1, dim + 1) * x, axis= -1)) ** 2 + (bm.sum(0.5 * bm.arange(1, dim + 1) * x, axis=-1)) ** 4

def F10(x):
    """
    Schwefel 2.22
    """
    return bm.sum(bm.abs(x), axis= -1) + bm.prod(bm.abs(x), axis= -1)

def F11(x):
    """
    Schwefel 1.2
    """
    dim = x.shape[-1]
    o = 0
    for j in range(1, dim + 1):
        oo = 0
        for k in range(1, j + 1):
            oo += x[:,k - 1]
        o += oo ** 2
    return o

def F12(x):
    """
    Dixon-Price
    """
    dim = x.shape[-1]
    o = (x[:,0] - 1) ** 2
    for j in range(2, dim + 1):
        o += j * (2 * x[:,j - 1] ** 2 - x[:,j - 1] - 1) ** 2
    return o

def F13(x):
    """
    Bohachevsky1
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) - 0.4 * bm.cos(4 * bm.pi * x[:,1]) + 0.7

def F14(x):
    """
    Booth
    """
    return (x[:,0] + 2 * x[:,1]) ** 2 + (2 * x[:,0] + x[:,1] - 5) ** 2

def F15(x):
    """
    Michalewicz2
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis= -1)  

def F16(x):
    """
    Michalewicz5
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis=-1)

def F17(x):
    """
    Michalewicz10
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis=-1)

def F18(x):
    """
    Rastrigin
    """
    n = x.shape[-1]
    o = 0
    for i in range(n):
        o += x[:,i] ** 2 - 10 * bm.cos(2 * bm.pi * x[:,i]) + 10
    return o

def F19(x):
    """
    Schaffer
    """
    return 0.5 + ((bm.sin(bm.linalg.norm(x, axis= -1))) ** 2 - 0.5) / ((1 + 0.001 * (bm.linalg.norm(x,axis=-1) ** 2)) ** 2)


def F20(x):
    """
    Six Hump Camel Back
    """
    return 4 * x[:,0] ** 2 - 2.1 * x[:,0] ** 4 + (1 / 3) * x[:,0] ** 6 + x[:,0] * x[:,1] - 4 * x[:,1] ** 2 + 4 * x[:,1] ** 4

def F21(x):
    """
    Boachevsky2
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) * bm.cos(4 * bm.pi * x[:,1]) + 0.3

def F22(x):
    """
    Boachevsky3
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) * bm.cos(4 * bm.pi * x[:,1]) + 0.3

def F23(x):
    """
    Shubert
    """
    x1 = x[:,0]
    x2 = x[:,1]
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 += i * bm.cos((i + 1) * x1 + i)
        sum2 += i * bm.cos((i + 1) * x2 + i)
    return sum1 * sum2

def F24(x):
    """
    Rosenbrock
    """
    o = 0
    n = x.shape[-1]
    for i in range(n - 1):
        o += 100 * (x[:,i + 1] - x[:,i] ** 2) ** 2 + (x[:,i] - 1) ** 2
    return o

def F25(x):
    """
    Griewank
    """
    n = x.shape[-1]
    y1 = bm.sum(x ** 2, axis =-1) / 4000
    y2 = 1
    for i in range(n):
        y2 *= bm.cos(x[:,i] / bm.sqrt(bm.array(i + 1)))
    return 1 + y1 - y2

def F26(x):
    """
    Ackley
    """
    y1 = bm.sum(x ** 2, axis= -1)
    y2 = bm.sum(bm.cos(2 * bm.pi * x), axis=-1 )
    n = len(x)
    return -20 * bm.exp(-0.2 * bm.sqrt(y1 / n)) - bm.exp(y2 / n) + 20 + bm.exp(bm.array(1))

iopt_benchmark_data = [
    {   
        "objective": F1, #Step
        "ndim": 30,
        "domain": (-5.12, 5.12), # (lower bound, higher bound)
        "minimum": -0.5*bm.ones(30), # 极小值0点
        "optimal": 0, # 极小值
    },
    {
        "objective": F2,#Sphere
        "ndim": 30,
        "domain": (-100, 100),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
    {
        "objective": F3,#Sum Squares
        "ndim": 30,
        "domain": (-10, 10),
        "minimum": None,
        "optimal": 0,
    },
    {
        "objective": F4,#Bent Cigar
        "ndim": 30,
        "domain": (-100, 100),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
    {
        "objective": F5,
        "ndim": 2,
        "domain": (-4.5, 4.5),
        "minimum": [3,0.5],
        "optimal": 0,
    },
    {
        "objective": F6,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": [0,0],
        "optimal": -1,
    },
    {
        "objective": F7,
        "ndim": 2,
        "domain": (-10, 10),
        "minimum": [0,0],
        "optimal": 0,
    },
    {
        "objective": F8,
        "ndim": 4,
        "domain": (-10, 10),
        "minimum": [1,1,1,1],
        "optimal": 0,
    },
    {
        "objective": F9,
        "ndim": 10,
        "domain": (-5, 10),
        "minimum": bm.zeros(10),
        "optimal": 0,
    },
    {
        "objective": F10,
        "ndim": 30,
        "domain": (-10, 10),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
    {
        "objective": F11,
        "ndim": 10,
        "domain": (-10, 10),
        "minimum": bm.zeros(10),
        "optimal": 0,
    },
    ##########
    {
        "objective": F12,
        "ndim": 30,
        "domain": (-10, 10),
        "minimum": None,
        "optimal": 0,
    },
    {
        "objective": F13,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": [0,0],
        "optimal": 0,
    },
    #########
    {
        "objective": F14,
        "ndim": 2,
        "domain": (-10, 10),
        "minimum": None,
        "optimal": 0,
    },
    {
        "objective": F15,
        "ndim": 2,
        "domain": (-bm.pi, bm.pi),
        "minimum": [0,0],
        "optimal": -1.8013,
    },
    #########
    {
        "objective": F16,
        "ndim": 5,
        "domain": (-bm.pi, bm.pi),
        "minimum": None,
        "optimal": -4.6877,
    },
    ######
    {
        "objective": F17,
        "ndim": 10,
        "domain": (-bm.pi, bm.pi),
        "minimum": None,
        "optimal": -9.6602,
    },
    {
        "objective": F18,
        "ndim": 30,
        "domain": (-5.12, 5.12),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
    {
        "objective": F19,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": [0,0],
        "optimal": 0,
    },
    {
        "objective": F20,
        "ndim": 2,
        "domain": (-5, 5),
        "minimum": [[-0.08984201368301331, 0.7126564032704135],[-0.08984201368301331, -0.7126564032704135]],#有两个x最优取值
        "optimal": -1.03163,
    },
    ###############
    {
        "objective": F21,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": None,
        "optimal": 0,
    },
    #############
    {
        "objective": F22,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": None,
        "optimal": 0,
    },
    {
        "objective": F23,
        "ndim": 5,
        "domain": (-10, 10),
        "minimum": [-7.0835,4.8580],#and many others
        "optimal": -186.7309,
    },
    {
        "objective": F24,
        "ndim": 30,
        "domain": (-30, 30),
        "minimum": bm.ones(30),
        "optimal": 0,
    },
    {
        "objective": F25,
        "ndim": 30,
        "domain": (-600, 600),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
    {
        "objective": F26,
        "ndim": 30,
        "domain": (-32, 32),
        "minimum": bm.zeros(30),
        "optimal": 0,
    },
]

Functions_dict = {
        'F2' : (F2, -100, 100, 30),
        'F3' : (F3, -10, 10, 30),
        'F4' : (F4, -100, 100, 30),
        'F5' : (F5, -4.5, 4.5, 2),
        'F6' : (F6, -100, 100, 2),
        'F7' : (F7, -10, 10, 2),
        'F8' : (F8, -10, 10, 4),
        'F9' : (F9, -5, 10, 10),
        'F10' : (F10, -10, 10, 30),
        'F11' : (F11, -10, 10, 10),
        'F12' : (F12, -10, 10, 30),
        'F13' : (F13, -100, 100, 2),
        'F14' : (F14, -10, 10, 2),
        'F15' : (F15, -bm.pi, bm.pi, 2),
        'F16' : (F16, -bm.pi, bm.pi, 5),
        'F17' : (F17, -bm.pi, bm.pi, 10),
        'F18' : (F18, -5.12, 5.12, 30),
        'F19' : (F19, -100, 100, 2),
        'F20' : (F20, -5, 5, 2),
        'F21' : (F21, -100, 100, 2),
        'F22' : (F22, -100, 100, 2),
        'F23' : (F23, -10, 10, 5),
        'F24' : (F24, -30, 30, 30),
        'F25' : (F25, -600, 600, 30),
        'F26' : (F26, -32, 32, 30),
    }