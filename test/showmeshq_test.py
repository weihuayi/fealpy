import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import show_mesh_quality



a = np.fromfile("../result/tet_opt_6_10970_50674_init.dat")
b = np.fromfile("../result/tet_opt_6_10970_50674_opt.dat")
a = np.fromfile("../result/hex_opt_6_7540_6328_init.dat")
b = np.fromfile("../result/hex_opt_6_7540_6328_opt.dat")
fig = plt.figure()
axes0 = fig.add_subplot(121)
show_mesh_quality(axes0, None, a)
axes1 = fig.add_subplot(122)
show_mesh_quality(axes1, None, b) 
ylim = axes1.get_ylim()
axes0.set_ylim(ylim[0], ylim[1])
plt.show()
