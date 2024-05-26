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