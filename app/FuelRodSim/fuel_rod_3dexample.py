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