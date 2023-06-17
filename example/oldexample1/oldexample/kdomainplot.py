import numpy as np
import matplotlib.pyplot as plt

node = np.array([(0, 0), (1, 0), (1, 1), (2,0),
    (2+np.sqrt(2), 0), (1.5, 2), (2+np.sqrt(2), 4), 
    (2, 4), (1, 3), (1, 4), (0,4)], dtype=np.float)

N = node.shape[0]
idx = np.repeat(range(N), 2)
edge = np.roll(idx, -1).reshape(-1, 2)

fig = plt.figure()
axes = fig.gca()
axes.plot(node[edge[:, 0], 0], node[edge[:, 0], 1])
axes.set_aspect('equal')
axes.set_axis_off()
plt.show()
