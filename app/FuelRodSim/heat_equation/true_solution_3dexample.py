# 三维带真解的测试
# TODO: 1. 三维真解的测试(其他五个已经通过)
from fealpy.mesh import TetrahedronMesh
from app.FuelRodSim.HeatEquationData import Parabolic3dData
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
pde=Parabolic3dData('sin(pi*x)*sin(pi*y)*sin(pi*z)*exp(-3*pi*t)','x','y','z','t')
nx = 5
ny = 5
nz = 5
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx, ny, nz)
node = mesh.node
isBdNode = mesh.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box3DSolver = HeatEquationSolver(mesh, pde, 160, isBdNode, p0=p0, alpha_caldding=1, layered=False, output='./result_box3dtest')
Box3DSolver.solve()
Box3DSolver.plot_exact_solution()
Box3DSolver.plot_error()
Box3DSolver.plot_error_over_time()