#使用示例，三维箱子热传导
if __name__ == "__main__":
    nx = 20
    ny = 20

from app.FuelRodSim.HeatEquationData import FuelRod2dData 
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
from fealpy.mesh import TriangleMesh
mesh = TriangleMesh.from_box([0, 1, 0, 1],nx,ny)
node = mesh.node
isBdNode = mesh.boundary_node_flag()
pde = FuelRod2dData()
Boxslover = HeatEquationSolver(mesh,pde,120,isBdNode,300,alpha_caldding=0.3,layered=False,output='./rusult_boxtest_2d')
Boxslover.solve()