
import numpy as np
from timeit import default_timer as dtimer 

N = 500
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start = dtimer()
C = A@B;
end = dtimer()
print(end - start)

start = dtimer()
D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        for k in range(N):
            D[i, j] += A[i, k]*B[k, j]

end = dtimer()
print(end - start)

