# 二维带真解的测试案例
from fealpy.mesh import TriangleMesh
from app.FuelRodSim.HeatEquationData import Parabolic2dData
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
pde=Parabolic2dData('exp(-2*pi**2*t)*sin(pi*x)*sin(pi*y)','x','y','t')
nx = 20
ny = 20
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx,ny)
node = mesh.node
print(node.shape)
isBdNode = mesh.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box2dsolver = HeatEquationSolver(mesh,pde,160,isBdNode,p0=p0,alpha_caldding=1,layered=False,output='./rusult_box2dtesttest_to')
Box2dsolver.solve()
Box2dsolver.plot_exact_solution() # 绘制真解
Box2dsolver.plot_error()
Box2dsolver.plot_exact_solution_heatmap() # 绘制真解
Box2dsolver.plot_error_heatmap()
Box2dsolver.plot_error_over_time() 