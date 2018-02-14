import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import show_mesh_quality


f = ['sphere/tet_opt_8_530_1777', 
        'gear/tet_opt_8_7677_34332',
        'lxy/tet_opt_8_16561_88544',
        'cad1/hex_opt_8_10181_8214',
        'cad2/hex_opt_8_2435_1876',
        'lyx_part/hex_opt_8_5628_4746'
        ]

i=0
for fi in f:
    a = np.fromfile("../result/"+fi+"_init.dat")
    b = np.fromfile("../result/"+fi+"_opt.dat")
    fig = plt.figure()
    axes0 = fig.add_subplot(121)
    show_mesh_quality(axes0, None, a)
    axes1 = fig.add_subplot(122)
    show_mesh_quality(axes1, None, b) 
    ylim = axes1.get_ylim()
    axes0.set_ylim(ylim[0], ylim[1])
    fig.savefig(str(i)+'.pdf')
    i += 1
