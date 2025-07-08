from fealpy.utils import timer
from fealpy import logger
logger.setLevel('WARNING')

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, UniformMesh2d
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.fem import InterfacePoissonLFEMModel
from fealpy.decorator import cartesian
from fealpy.solver import cg

def interface(p):
    x = p[..., 0]
    y = p[..., 1]
    return x**2 + y**2 - 2.1**2

class interfacefunction:
    def __init__(self):
        self.beta = 1

    def domain(self):
        return [-5, 5, -5, 5]

    @cartesian
    def solution(self, p):
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]

        val = bm.zeros(x.shape, **kwargs)

        Omega0 = interface(p) >0 
        Omega1 = interface(p) <0

        val[Omega0] = -x[Omega0] + 2.1**2*x[Omega0]/(x[Omega0]**2 + y[Omega0]**2)
        val[Omega1] = 0

        return val

    @cartesian
    def source(self, p):
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        val = bm.zeros(x.shape, **kwargs)

        return val 

    @cartesian
    def flux(self, p, n):
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        n2 = bm.zeros(x.shape, **kwargs)
        n2 = -(2*x)/2.1

        return n2


p=1
pde = interfacefunction()  
domain = pde.domain()

nx = 20
ny = 20

hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

tmr = timer()
next(tmr)
back_mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))
iCellNodeIndex, cutNode, auxNode, isInterfaceCell = back_mesh.find_interface_node(interface)
mesh = TriangleMesh.interfacemesh_generator(back_mesh, interface)
tmr.send('网格和界面拟合时间')

maxit = 5
em = bm.zeros((2, maxit), dtype=bm.float64)
for i in range(maxit):
    space= LagrangeFESpace(mesh, p=p)
    tmr.send(f'第{i}次空间时间')

    uh = space.function()

    model = InterfacePoissonLFEMModel(mesh, space, pde, interface)
    tmr.send(f'第{i}生成模型时间')

    A, f = model.linear_system()
    tmr.send(f'第{i}次矩组装时间')

    uh[:] = cg(A, f, atol=1e-14, rtol=1e-14)
    tmr.send(f'第{i}次求解器时间')

    em[0, i] = mesh.error(uh, pde.solution, power=1)
    em[1, i] = mesh.error(uh, pde.solution, power=2)

    if i < maxit:
        back_mesh.uniform_refine()
        iCellNodeIndex, cutNode, auxNode, isInterfaceCell = back_mesh.find_interface_node(interface)
        mesh = TriangleMesh.interfacemesh_generator(back_mesh, interface)
    tmr.send(f'第{i}次误差计算及网格加密时间')
next(tmr)
print("最终误差", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])