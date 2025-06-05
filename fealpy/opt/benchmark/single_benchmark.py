from fealpy.backend import backend_manager as bm

def F1(x):
    """
    Step

    This function calculates the Step function, which is the sum of the squares of the input vector plus 0.5.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed sum of squared values plus 0.5.
    """
    return bm.sum((x + 0.5) ** 2, axis=-1)

def F2(x):
    """
    Sphere

    This function computes the Sphere function, which is the sum of the squares of the input tensor.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed sum of squared values.
    """
    return bm.sum(x ** 2, axis=-1)

def F3(x):
    """
    Sum Squares

    This function calculates the sum of squares of the input tensor, weighted by increasing factors.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed sum of weighted squares.
    """
    dim = x.shape[-1]
    return bm.sum(bm.arange(1, dim + 1) * (x ** 2), axis=-1)

def F4(x):
    """
    Bent Cigar

    This function calculates the Bent Cigar function, which is the sum of a large penalty for the non-first dimensions.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Bent Cigar function.
    """
    return x[:,0] ** 2 + bm.sum(1e6 * x[:,1:] ** 2, axis=-1)

def F5(x):
    """
    Beale

    This function computes the Beale function, a well-known optimization test function.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Beale function.
    """
    return (1.5 - x[:,0] + x[:,0] * x[:,1]) ** 2 + (2.25 - x[:,0] + x[:,0] * x[:,1] ** 2) ** 2 + (2.625 - x[:,0] + x[:,0] * x[:,1] ** 3) ** 2

def F6(x):
    """
    Easom

    This function calculates the Easom function, which is used in optimization benchmarks.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Easom function.
    """
    return -bm.cos(x[:,0]) * bm.cos(x[:,1]) * bm.exp(-((x[:,0] - bm.pi) ** 2 + (x[:,1] - bm.pi) ** 2))

def F7(x):
    """
    Matyas

    This function computes the Matyas function, a simple quadratic function used for optimization testing.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Matyas function.
    """
    return 0.26 * (x[:,0] ** 2 + x[:,1] ** 2) - 0.48 * x[:,0] * x[:,1]

def F8(x):
    """
    Colville

    This function calculates the Colville function, a benchmark function in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Colville function.
    """
    return 100 * (x[:,0] ** 2 - x[:,1]) ** 2 + (x[:,0] - 1) ** 2 + (x[:,2] - 1) ** 2 + 90 * (x[:,2] ** 2 - x[:,3]) ** 2 + 10.1 * ((x[:,1] - 1) ** 2 + (x[:,3] - 1) ** 2) + 19.8 * (x[:,1] - 1) * (x[:,3] - 1)

def F9(x):
    """
    Zakharov

    This function computes the Zakharov function, an optimization benchmark.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Zakharov function.
    """
    dim = x.shape[-1]
    return bm.sum(x ** 2, axis= -1) + (bm.sum(0.5 * bm.arange(1, dim + 1) * x, axis= -1)) ** 2 + (bm.sum(0.5 * bm.arange(1, dim + 1) * x, axis=-1)) ** 4

def F10(x):
    """
    Schwefel 2.22

    This function calculates the Schwefel 2.22 function, often used for optimization testing.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Schwefel 2.22 function.
    """
    return bm.sum(bm.abs(x), axis= -1) + bm.prod(bm.abs(x), axis= -1)

def F11(x):
    """
    Schwefel 1.2

    This function computes the Schwefel 1.2 function, a benchmark function in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Schwefel 1.2 function.
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

    This function calculates the Dixon-Price function, used in optimization benchmarks.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Dixon-Price function.
    """
    dim = x.shape[-1]
    o = (x[:,0] - 1) ** 2
    for j in range(2, dim + 1):
        o += j * (2 * x[:,j - 1] ** 2 - x[:,j - 1] - 1) ** 2
    return o

def F13(x):
    """
    Bohachevsky1

    This function computes the Bohachevsky1 function, a benchmark function in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Bohachevsky1 function.
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) - 0.4 * bm.cos(4 * bm.pi * x[:,1]) + 0.7

def F14(x):
    """
    Booth

    This function calculates the Booth function, a classic test problem in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Booth function.
    """
    return (x[:,0] + 2 * x[:,1]) ** 2 + (2 * x[:,0] + x[:,1] - 5) ** 2

def F15(x):
    """
    Michalewicz2

    This function computes the Michalewicz2 function, a commonly used benchmark in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Michalewicz2 function.
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis= -1)

def F16(x):
    """
    Michalewicz5

    This function computes the Michalewicz5 function, a commonly used benchmark in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Michalewicz5 function.
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis=-1)

def F17(x):
    """
    Michalewicz10

    This function computes the Michalewicz10 function, a commonly used benchmark in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Michalewicz10 function.
    """
    dim = x.shape[-1]
    return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20), axis=-1)

def F18(x):
    """
    Rastrigin

    This function computes the Rastrigin function, a benchmark function often used in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Rastrigin function.
    """
    n = x.shape[-1]
    o = 0
    for i in range(n):
        o += x[:,i] ** 2 - 10 * bm.cos(2 * bm.pi * x[:,i]) + 10
    return o

def F19(x):
    """
    Schaffer

    This function computes the Schaffer function, a popular benchmark in optimization.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Schaffer function.
    """
    return 0.5 + (bm.sin(x[:,0] ** 2 - x[:,1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[:,0] ** 2 + x[:,1] ** 2)) ** 2

def F20(x):
    """
    Rosenbrock

    This function computes the Rosenbrock function, a well-known optimization test problem.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Rosenbrock function.
    """
    return bm.sum(100 * (x[:,1:] - x[:,:-1] ** 2) ** 2 + (1 - x[:,:-1]) ** 2, axis=-1)

def F21(x):
    """
    Boachevsky2

    This function computes the Boachevsky2 function, a benchmark function in optimization, with a combination of quadratic terms and trigonometric components.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Boachevsky2 function.
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) * bm.cos(4 * bm.pi * x[:,1]) + 0.3

def F22(x):
    """
    Boachevsky3

    This function calculates the Boachevsky3 function, a variant of the Boachevsky function with more complex trigonometric interactions.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Boachevsky3 function.
    """
    return x[:,0] ** 2 + 2 * x[:,1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[:,0]) * bm.cos(4 * bm.pi * x[:,1]) + 0.3

def F23(x):
    """
    Shubert

    This function computes the Shubert function, a multimodal function with multiple local minima, used in optimization benchmarks.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Shubert function.
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

    This function calculates the Rosenbrock function, a classic optimization test problem involving a sum of squared differences.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Rosenbrock function.
    """
    o = 0
    n = x.shape[-1]
    for i in range(n - 1):
        o += 100 * (x[:,i + 1] - x[:,i] ** 2) ** 2 + (x[:,i] - 1) ** 2
    return o

def F25(x):
    """
    Griewank

    This function computes the Griewank function, an optimization benchmark that involves both sum of squares and product of cosines.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Griewank function.
    """
    n = x.shape[-1]
    y1 = bm.sum(x ** 2, axis =-1) / 4000
    y2 = bm.prod(bm.cos(x / bm.sqrt(bm.arange(1, n+1))), axis=-1)
    return 1 + y1 - y2

def F26(x):
    """
    Ackley

    This function calculates the Ackley function, an optimization benchmark with a characteristic large flat region and steep peaks.

    Parameters:
        x (Tensor): Input tensor.

    Returns:
        Tensor: The computed Ackley function.
    """
    y1 = bm.sum(x ** 2, axis= -1)
    y2 = bm.sum(bm.cos(2 * bm.pi * x), axis=-1)
    n = x.shape[-1]
    return -20 * bm.exp(-0.2 * bm.sqrt(y1 / n)) - bm.exp(y2 / n) + 20 + bm.exp(bm.array(1))


single_benchmark_data = [
    {   
        "objective": F1, #Step
        "ndim": 30,
        "domain": (-5.12, 5.12), # (lower bound, higher bound)
        "minimum": -0.5 * bm.ones(30), # 极小值0点
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
        "minimum": [3, 0.5],
        "optimal": 0,
    },
    {
        "objective": F6,
        "ndim": 2,
        "domain": (-100, 100),
        "minimum": [0, 0],
        "optimal": -1,
    },
    {
        "objective": F7,
        "ndim": 2,
        "domain": (-10, 10),
        "minimum": [0, 0],
        "optimal": 0,
    },
    {
        "objective": F8,
        "ndim": 4,
        "domain": (-10, 10),
        "minimum": [1, 1, 1, 1],
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
        "minimum": [0, 0],
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
        "minimum": [0, 0],
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
        "minimum": [0, 0],
        "optimal": 0,
    },
    {
        "objective": F20,
        "ndim": 2,
        "domain": (-5, 5),
        "minimum": [[-0.08984201368301331, 0.7126564032704135], [-0.08984201368301331, -0.7126564032704135]],
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
        "minimum": [-7.0835, 4.8580],
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