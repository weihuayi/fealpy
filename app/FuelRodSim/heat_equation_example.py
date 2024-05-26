"""
# 使用示例:二维的燃料棒slover
if __name__ == "__main__":
    mm = 1e-03
    #包壳厚度
    w = 0.15 * mm
    #半圆半径
    R1 = 0.5 * mm
    #四分之一圆半径
    R2 = 1.0 * mm
    #连接处直线段
    L = 0.575 * mm
    #内部单元大小
    h = 0.5 * mm
    #棒长
    l = 20 * mm
    #螺距
    p = 40 * mm

from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher
from app.FuelRodSim.HeatEquationData import FuelRod2dData 
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
mesher = FuelRodMesher(R1,R2,L,w,h,meshtype='segmented',modeltype='2D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_2D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_2D_cnidx_bdnidx()
pde = FuelRod2dData()
FuelRodsolver = HeatEquationSolver(mesh, pde,640, bdnidx,300,ficdx=ficdx,cacidx=cacidx,output='./result_fuelrod2Dtest')
FuelRodsolver.solve()
"""

"""
# 使用示例:三维的燃料棒slover
if __name__ == "__main__":
    mm = 1e-03
    #包壳厚度
    w = 0.15 * mm
    #半圆半径
    R1 = 0.5 * mm
    #四分之一圆半径
    R2 = 1.0 * mm
    #连接处直线段
    L = 0.575 * mm
    #内部单元大小
    h = 0.5 * mm
    #棒长
    l = 20 * mm
    #螺距
    p = 40 * mm
from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher
from app.FuelRodSim.HeatEquationData import FuelRod3dData 
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
mesher = FuelRodMesher(R1,R2,L,w,h,l,p,meshtype='segmented',modeltype='3D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_3D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_3D_cnidx_bdnidx()
pde = FuelRod3dData()
FuelRodsolver = HeatEquationSolver(mesh, pde,64, bdnidx,300,ficdx=ficdx,cacidx=cacidx,output='./result_fuelrod3Dtest')
FuelRodsolver.solve()
"""

"""
#使用示例，三维箱子热传导
if __name__ == "__main__":
    nx = 20
    ny = 20
    nz = 20

from app.FuelRodSim.HeatEquationData import FuelRod3dData 
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
from fealpy.mesh import TetrahedronMesh
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1],nx,ny,nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
pde = FuelRod3dData()
Boxslover = HeatEquationSolver(mesh,pde,120,isBdNode,300,alpha_caldding=0.08,layered=False,output='./rusult_boxtest')
Boxslover.solve()
"""

"""
# 二维带真解的测试案例
from fealpy.mesh import TriangleMesh
from app.FuelRodSim.HeatEquationData import Parabolic2dData
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
pde=Parabolic2dData()
nx = 20
ny = 20
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx,ny)
node = mesh.node
print(node.shape)
isBdNode = mesh.ds.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box2dslover = HeatEquationSolver(mesh,pde,160,isBdNode,p0=p0,alpha_caldding=1,layered=False,output='./rusult_box2dtesttest')
Box2dslover.solve()
Box2dslover.plot_exact_solution() # 绘制真解
Box2dslover.plot_error()
"""


# 三维带真解的测试
from fealpy.mesh import TetrahedronMesh
from app.FuelRodSim.HeatEquationData import Parabolic3dData
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
pde=Parabolic3dData()
nx = 5
ny = 5
nz = 5
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx, ny, nz)
node = mesh.node
isBdNode = mesh.ds.boundary_node_flag()
p0=pde.init_solution(node) #准备一个初值
Box3DSolver = HeatEquationSolver(mesh, pde, 160, isBdNode, p0=p0, alpha_caldding=1, layered=False, output='./result_box3dtest')
Box3DSolver.solve()
Box3DSolver.plot_exact_solution()
Box3DSolver.plot_error()

