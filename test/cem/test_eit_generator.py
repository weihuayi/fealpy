
from fealpy.cem.generator import EITDataGenerator
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, UniformMesh2d


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

umesh = UniformMesh2d([0, EXT, 0, EXT], [2./EXT,]*2, [-1, -1])
mesh = TriangleMesh.interfacemesh_generator(umesh, phi=inclusion)

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