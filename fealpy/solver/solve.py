
import numpy as np
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from scipy.sparse import spdiags
from timeit import default_timer as timer
#import pyamg
import pylab

def solve1(a, L, uh, dirichlet=None, neuman=None, solver='cg'):
    V = a.V

    start = timer()
    A = a.get_matrix()
    b = L.get_vector()
    end = timer()

    print("Construct linear system time:", end - start)

    if neuman is not None:
        b += neuman.get_vector()

    if dirichlet is not None:
        AD, b = dirichlet.apply(A, b)

#    print("The condtion number is: ", np.linalg.cond(AD.todense()))
    if solver is 'cg':
        start = timer()
        D = AD.diagonal()
        M = spdiags(1/D, 0, AD.shape[0], AD.shape[1])
        uh[:], info = cg(AD, b, tol=1e-14, M=M)
        end = timer()
        print(info)
    elif solver is 'amg':
        start = timer()
        ml = pyamg.ruge_stuben_solver(AD)  
        uh[:] = ml.solve(b, tol=1e-12, accel='cg').reshape((-1,))
        end = timer()
        print(ml)
    elif solver is 'direct':
        start = timer()
        uh[:] = spsolve(AD, b)
        end = timer()
    else:
        print("We don't support solver: " + solver)

    print("Solve time:", end-start)

    return A 

def solve(fem, uh, dirichlet=None, solver='cg'):
    V = fem.V
    start = timer()
    A = fem.get_left_matrix()
    b = fem.get_right_vector()
    end = timer()

    print("Construct linear system time:", end - start)

    if dirichlet is not None:
        AD, b = dirichlet.apply(A, b)

    if solver is 'cg':
        start = timer()
        D = AD.diagonal()
        M = spdiags(1/D, 0, AD.shape[0], AD.shape[1])
        uh[:], info = cg(AD, b, tol=1e-14, M=M)
        end = timer()
        print(info)
    elif solver is 'amg':
        start = timer()
        ml = pyamg.ruge_stuben_solver(AD)  
        uh[:] = ml.solve(b, tol=1e-12, accel='cg').reshape((-1,))
        end = timer()
        print(ml)
    elif solver is 'direct':
        start = timer()
        uh[:] = spsolve(AD, b)
        end = timer()
    else:
        raise ValueError("We don't support solver `{}`! ".format(solver))

    print("Solve time:", end-start)

    return A, b 
