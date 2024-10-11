
from fealpy.cem.generator import EITDataGenerator
from fealpy.mesh import TriangleMesh as TM
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh


EXT = 63
bm.set_backend('pytorch')

def inclusion(p):
    x = p[..., 0]
    y = p[..., 1]
    return (x-0.5)**2 + (y-0.5)**2 - 0.16
inclusion.coordtype = 'cartesian'

def current_density(p, *args):
    angle = bm.atan2(p[..., 1], p[..., 0])
    return bm.sin(1*angle)
current_density.coordtype = 'cartesian'

mesh = TM.interfacemesh_generator([-1, 1, -1, 1], nx=EXT, ny=EXT, phi=inclusion)
mesh = TriangleMesh(bm.from_numpy(mesh.entity('node')),
                    bm.from_numpy(mesh.entity('cell')))

gen = EITDataGenerator(mesh, p=1)
flag = gen.set_levelset([10., 1.], inclusion)
current = gen.set_boundary(current_density)
uh = gen.run(return_full=True)
print("Shape of uh: ", uh.shape)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_node(axes, color=bm.to_numpy(uh))
    plt.show()