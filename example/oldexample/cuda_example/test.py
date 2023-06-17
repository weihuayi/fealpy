import numpy as np
import cupy as cp
from cupy import cutensor
from time import perf_counter_ns

def generateTestDataDim(p, q, numberOfCell, geometryDim=3, topologyDim=3):
    I = q * (q+1) * (q+2) / 6
    K = M = (p+1) * (p+2) * (p+3) / 6
    J = numberOfCell
    L = geometryDim
    return list(map(int, (I, J, K, L, M)))

dtype = np.float32

I, J, K, L, M = generateTestDataDim(3, 3, 100000)
print("Data benchmark:[I,J,K,L,M]=[{},{},{},{},{}]\n".format(I, J, K, L, M))
a = np.random.rand(I).astype(dtype)
b = np.random.rand(I, J, K, L).astype(dtype)
c = np.random.rand(I, J, M, L).astype(dtype)
d = np.random.rand(J).astype(dtype)

warmup=1
cycle=5

# for plot
time_data=[]
method_name=[]
acc=[]

print("Method: numpy.einsum (baseline)")
t_avg=0.0
for i in range(2):
    t1_start = perf_counter_ns()
    e0 = np.einsum('i, ijkl, ijml, j->jkm', a, b, c, d)
    t1_stop = perf_counter_ns()
    t1 = (t1_stop-t1_start) / 1e6
    print(f"Cycle {i+1}: {t1:5.5f} ms.")
    if i >= warmup:
        t_avg += t1
time = t_avg/(1)
print(f"Average time: {time:5.5f} ms")
baseline_time = time

time_data.append(time)
method_name.append("numpy.einsum")
acc.append(np.nan)

print("Method: cupy.einsum")

a_gpu = cp.asarray(a)
b_gpu = cp.asarray(b)
c_gpu = cp.asarray(c)
d_gpu = cp.asarray(d)

t_avg=0
for i in range(cycle):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e_gpu = cp.empty((J, K, M)).astype(dtype)
    e_gpu[:, :, :] = cp.einsum(
        'i, ijkl, ijml, j->jkm', a_gpu, b_gpu, c_gpu, d_gpu)
    e1 = e_gpu.get()
    cp.cuda.Stream.null.synchronize()
    t1_stop = perf_counter_ns()
    assert(np.allclose(e1,e0))
    t1 = (t1_stop-t1_start) / 1e6
    print(f"Cycle {i+1}: {t1:5.5f} ms.")
    if i >= warmup:
        t_avg += t1
time = t_avg/(cycle-warmup)
acc_rate = baseline_time/time
print(f"Average time: {time:5.5f} ms")
print(f"Accelerate  : {acc_rate:5.5}")

time_data.append(time)
method_name.append("cupy.einsum")
acc.append(acc_rate)


