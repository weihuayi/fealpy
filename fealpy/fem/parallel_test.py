import numpy as np
from scipy.sparse import csr_matrix
from mumps import spsolve
from ..solver import solve

AA = np.loadtxt('AA.txt')

JA = np.loadtxt('JA.txt')
JA = JA - 1

IA = np.loadtxt('IA.txt')
IA = IA - 1

F = np.loadtxt('f.txt')

u = np.loadtxt('solution.txt')

n = len(F)
A = csr_matrix((AA, JA, IA), shape=(n, n))
uh = np.zeros(n, dtype=np.float)
#uh[:] = spsolve(A, F)
solve1(A, F, uh, solver='cg')
print(uh)
e = uh - u
get_l2_error = np.sqrt(np.mean(e**2))
print(get_l2_error)
