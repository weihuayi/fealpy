import os

import scipy.io as sio
from fealpy.backend import backend_manager as bm

def get_PF(function_name):
    """
    Load the true Pareto Front (PF) from a .mat file.
    
    Args:
        function_name (str): Name of the .mat file (e.g., 'ZDT1.mat', 'Poloni.mat').
    
    Returns:
        numpy.ndarray: The loaded Pareto Front points.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    mat_file_path = os.path.join(script_dir, function_name)
    pf = sio.loadmat(mat_file_path)
    PF = pf['PF']
    return PF

def Kursawe_F1(x):
    """
    Compute the first objective of the Kursawe problem.
    
    Args:
        x (bm.array): Input array of shape (n_samples, 3).
    
    Returns:
        bm.array: Values of the first objective.
    """
    a1 = bm.exp(-0.2 * bm.sqrt(x[:, 0]**2 + x[:, 1]**2))
    a2 = bm.exp(-0.2 * bm.sqrt(x[:, 1]**2 + x[:, 2]**2))
    return -10 * (a1 + a2)

def Kursawe_F2(x):
    """
    Compute the second objective of the Kursawe problem.
    
    Args:
        x (bm.array): Input array of shape (n_samples, 3).
    
    Returns:
        bm.array: Values of the second objective.
    """
    return bm.sum(bm.abs(x)**0.8 + 5 * bm.sin(x**3), axis=-1)

def Poloni_f1(x, y):
    """
    Compute the first objective of the Poloni problem.
    
    Args:
        x (bm.array): First input variable.
        y (bm.array): Second input variable.
    
    Returns:
        bm.array: Values of the first objective.
    """
    def B1(x, y):
        return 0.5 * bm.sin(x) - 2 * bm.cos(x) + bm.sin(y) - 1.5 * bm.cos(y)
    
    def B2(x, y):
        return 1.5 * bm.sin(x) - bm.cos(x) + 2 * bm.sin(y) - 0.5 * bm.cos(y)
    
    A1 = 0.5 * bm.sin(1) - 2 * bm.cos(1) + bm.sin(2) - 1.5 * bm.cos(2)
    A2 = 1.5 * bm.sin(1) - bm.cos(1) + 2 * bm.sin(2) - 0.5 * bm.cos(2)
    return 1 + (A1 - B1(x, y))**2 + (A2 - B2(x, y))**2

def Poloni_f2(x, y):
    """
    Compute the second objective of the Poloni problem.
    
    Args:
        x (bm.array): First input variable.
        y (bm.array): Second input variable.
    
    Returns:
        bm.array: Values of the second objective.
    """
    return (x + 3)**2 + (y + 1)**2

def ZDT1_f1(x):
    """
    Compute the first objective of the ZDT1 problem.
    
    Args:
        x (bm.array): Input array of shape (n_samples, n_features).
    
    Returns:
        bm.array: Values of the first objective.
    """
    return x[:, 0]

def ZDT1_f2(x):
    """
    Compute the second objective of the ZDT1 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the second objective.
    """
    def g(x):
        return 1 + 9 * bm.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    
    fx = ZDT1_f1(x)
    gx = g(x)
    return gx * (1 - bm.sqrt(fx / gx))

def ZDT2_f1(x):
    """
    Compute the first objective of the ZDT2 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the first objective.
    """
    return x[:, 0]

def ZDT2_f2(x):
    """
    Compute the second objective of the ZDT2 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the second objective.
    """
    def g(x):
        return 1 + 9 * bm.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    
    def h(x):
        fx = ZDT2_f1(x)
        gx = g(x)
        return 1 - (fx / gx)**2
    
    return g(x) * h(x)

def ZDT3_f1(x):
    """
    Compute the first objective of the ZDT3 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the first objective.
    """
    return x[:, 0]

def ZDT3_f2(x):
    """
    Compute the second objective of the ZDT3 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the second objective.
    """
    def g(x):
        return 1 + 9 * bm.sum(x[:, 1:], axis=1) / (x.shape[1] - 1)
    
    def h(x):
        fx = ZDT3_f1(x)
        gx = g(x)
        return 1 - bm.sqrt(fx / gx) - (fx / gx) * bm.sin(10 * bm.pi * fx)
    
    return g(x) * h(x)

def ZDT6_f1(x):
    """
    Compute the first objective of the ZDT6 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the first objective.
    """
    return 1 - bm.exp(-4 * x[:, 0]) * bm.sin(6 * bm.pi * x[:, 0])

def ZDT6_f2(x):
    """
    Compute the second objective of the ZDT6 problem.
    
    Args:
        x (bm.array): Input array.
    
    Returns:
        bm.array: Values of the second objective.
    """
    def g(x):
        return 1 + 9 * (bm.sum(x[:, 1:], axis=1) / (x.shape[1] - 1))**0.25
    
    def h(x):
        return 1 - (ZDT6_f1(x) / g(x))**2
    
    return g(x) * h(x)


multi_benchmark_data = [
    {
        'fun': lambda x: bm.stack([x**2, (x-2)**2]).squeeze(-1).T,
        'ndim': 1,
        'lb': -5 * bm.ones((1)),
        'ub': 5 * bm.ones((1)),
        'PF': get_PF('Schaffer'),
    },
    {
        'fun': lambda x: bm.stack([Kursawe_F1(x), Kursawe_F2(x)]).T,
        'ndim': 3,
        'lb': -5 * bm.ones((3)),
        'ub': 5 * bm.ones((3)),
        'PF': get_PF('Kursawe'),
    },
    {
        'fun': lambda x: bm.stack([Poloni_f1(x[:, 0], x[:, 1]), Poloni_f2(x[:, 0], x[:, 1])], axis=1),
        'ndim': 2,
        'lb': -bm.pi * bm.ones((2,)),
        'ub': bm.pi * bm.ones((2,)),
        'PF': get_PF('Poloni'),
    },
    {
        'fun': lambda x: bm.stack([ZDT1_f1(x), ZDT1_f2(x)], axis=1),
        'ndim': 30,
        'lb': bm.zeros((30,)),
        'ub': bm.ones((30,)),
        'PF': get_PF('ZDT1'),
    },
    {
        'fun': lambda x: bm.stack([ZDT2_f1(x), ZDT2_f2(x)], axis=1),
        'ndim': 30,
        'lb': bm.zeros((30,)),
        'ub': bm.ones((30,)),
        'PF': get_PF('ZDT2'),
    },
    {
        'fun': lambda x: bm.stack([ZDT3_f1(x), ZDT3_f2(x)], axis=1),
        'ndim': 5,
        'lb': bm.zeros((5,)),
        'ub': bm.ones((5,)),
        'PF': get_PF('ZDT3'),
    },
    {
        'fun': lambda x: bm.stack([ZDT6_f1(x), ZDT6_f2(x)], axis=1),
        'ndim': 10,
        'lb': bm.zeros((10,)),
        'ub': bm.ones((10,)),
        'PF': get_PF('ZDT6'),
    },
]