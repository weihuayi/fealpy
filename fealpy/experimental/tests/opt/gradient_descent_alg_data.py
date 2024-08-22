import numpy as np

init_data = [
    {
        "x0" : np.array((0.0,0.0),dtype= np.float64),
        "objective":  lambda p : (p[0]**2 + p[1]**2 , np.array([2*p[0],2*p[1]])),
        "domain": np.array([0,1,0,1],dtype= np.float64),
        "NP" :  1,
        "MaxIters" : 1000,
        "MaxFunEvals" : 10000,
        "NormGradTol" :  1e-6,
        "FunValDiff":  1e-6,
        "StepLength":  1.0,
        "StepLengthTol":  1e-8,
        "NumGrad": 10

    }
]

run_data = [
    {
        "x0" : np.array((2.0,2.0),dtype= np.float64),
        "objective":  lambda p : (p[0]**2 + p[1]**2 , np.array([2*p[0],2*p[1]])),
        "StepLength": 0.4,
        "MaxIters" : 2000,
        "x" : np.array((0.0,0.0),dtype= np.float64),
        "f" : 0,
        "g" : np.array([0.0,0.0],np.float64),
        "diff" : 0
    },
    {
        "x0" : np.array((0.0,0.0),dtype= np.float64),
        "objective":  lambda p : ((p[0]-1)**2 + (p[1]-2)**2 , np.array([2*(p[0]-1),2*(p[1]-2)])),
        "StepLength": 0.4,
        "MaxIters" : 2000,
        "x" : np.array([1.0,2.0],dtype= np.float64),
        "f" : 0.,
        "g" : np.array([0.0,0.0],dtype= np.float64),
        "diff" : 0.
    }
]


