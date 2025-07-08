from fealpy.pde.parabolic2d import Parabolic2dData
from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
import os
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, BilinearForm, ScalarMassIntegrator, LinearForm, ScalarSourceIntegrator
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.solver import cg

#bm.set_backend('pytorch') # 选择后端为pytorch

pde=Parabolic2dData('exp(-2*pi**2*t)*sin(pi*x)*sin(pi*y)','x','y','t')
nx = 20
ny = 20
#mesh = TriangleMesh.from_box([0, 1, 0, 1], nx,ny)
mesh =TriangleMesh.from_unit_circle_gmsh(0.05)
node = mesh.node
isBdNode = mesh.boundary_node_flag()
p0 = pde.init_solution(node) #准备一个初值
p = bm.array(p0)

space = LagrangeFESpace(mesh, p=1)
GD = space.geo_dimension()
duration = pde.duration()
nt = 640
tau = (duration[1] -duration[0]) / nt


alpha = 0.5
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(alpha, q=3))
K = bform.assembly()

bform2 = BilinearForm(space)
bform2.add_integrator(ScalarMassIntegrator(q=3))
M = bform2.assembly()

# 当前文件夹下有没有该文件，没有则生成
import os
output = './result_truesolution_2d'
filename = 'temp'
if not os.path.exists(output):
    os.makedirs(output)
    
for n in range(nt):
    t = duration[0] + n * tau
    # 由于PDE模型基于符号计算，需要定义一个在笛卡尔坐标下的函数
    bform3 = LinearForm(space)
    from fealpy.decorator import cartesian
    @cartesian
    def coef(p):
        time = t
        val = pde.source(p, time)
        return val
    bform3.add_integrator(ScalarSourceIntegrator(coef))
    F = bform3.assembly()
    A = M +  K * tau
    b = M @ p + tau * F
    bc = DirichletBC(space=space,  gd=lambda p: pde.dirichlet(p,t))
    A, b = bc.apply(A, b)
    p = cg(A, b, maxit=5000, atol=1e-14, rtol=1e-14)
    # 生成vtu文件
    mesh.nodedata['temp'] = p.flatten()
    name = os.path.join(output, f'{filename}_{n:010}.vtu')
    mesh.to_vtk(fname=name)