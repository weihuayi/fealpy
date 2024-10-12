from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt.Levy import levy
import time
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
bm.set_backend('pytorch')

###################################
# import torch as bm
# device = bm.device('cuda' if bm.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
##############################

start_time = time.perf_counter()

lb = -100
ub = 100
n = 512 * 512 * 512
L = bm.zeros([n, 2]) + 10
B = bm.zeros([n, 2]) + 10

levy_steps = 0.01 * levy(n - 1, 2, 1.5)
randn_steps = bm.random.randn(n - 1, 2)

L[1:, :] = bm.cumsum(bm.array(levy_steps), axis=0) + 10
B[1:, :] = bm.cumsum(randn_steps, axis=0) + 10
x = L[:, 0]
y = L[:, 1]
xx = B[:, 0]
yy = B[:, 1]

end_time = time.perf_counter()
running_time = end_time - start_time
print("Running time:", running_time)

# plt.scatter(L[0, 0], L[0, 1], marker = MarkerStyle('*'), color = 'red', s = 500)
# plt.plot(x,y,'-', color = 'blue')
# plt.plot(xx, yy, '--', color = 'black')

# plt.show()