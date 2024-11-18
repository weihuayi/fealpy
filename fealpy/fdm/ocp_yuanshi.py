#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ocp-opt.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Sep 2024 04:50:58 PM CST
	@bref 
	@ref 
'''  
from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import TensorFunctionSpace
from ocp_opt_pde import example_1
from solver import ocp_opt_solver

from scipy.sparse import coo_array, bmat
from functools import partial
from fealpy import logger
logger.setLevel('ERROR') #积分子问题

bm.set_backend("numpy")
pde = example_1()
n = 10
q = 4
T = 1
nt = 30
maxit = 3

mesh = TriangleMesh.from_box(pde.domain(), nx=n, ny=n)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

yspace= LagrangeFESpace(mesh, p=1) 
space = LagrangeFESpace(mesh, p=1)
pspace = TensorFunctionSpace(space, (2,-1)) 
solver = ocp_opt_solver(mesh, yspace, pspace, pde, timeline)

ygodf = yspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
yisbdof = yspace.is_boundary_dof()
pisbdof = pspace.is_boundary_dof()
isbdof = bm.concatenate([yisbdof, pisbdof], axis=0)#(362,)


ally = [None]*(nt+1)
allp = [None]*(nt+1)
allu = [None]*(nt+1)


y0 = yspace.function(yspace.interpolate(partial(pde.y_solution, time=0)))
y0t = yspace.interpolate(partial(pde.y_t_solution, time=0)) 
p0 = pspace.function(pspace.interpolate(partial(pde.p_solution, time=0))) #(242,) p0x1,p0x2
ally[0] = y0
allp[0] = p0


for k in range(maxit):
    A0 = solver.A0n()
    b0 = solver.forward_b0(allu[1])
    A0, b0 = solver.forward_boundary(A0, b0, isbdof, dt)
    x0 = solver.mumps_solve(A0, b0)

    y1 = yspace.function()
    p1 = pspace.function()

    y1[:] = x0[:ygodf]
    p1[:] = x0[-pgdof:] #p1x1, p1x2
    ally[1] = y1
    allp[1] = p1
    timeline.advance()

    AA = solver.A()
    #正向求解
    for i in range(nt-1):
        t1 = timeline.next_time_level()
        tnextindex = timeline.current_time_level_index()+1 # 为什么要加1

        y2 = yspace.function()
        px = pspace.function()
        b = solver.forward_b(y0, y1, allu[tnextindex], t1)
        A,b = solver.forward_boundary(AA, b, isbdof, t1)
        
        x = solver.mumps_solve(A, b)
        y0[:] = y1
        y1[:] = x[:ygodf]
        px[:] = x[-pgdof:]
        ally[tnextindex] = y1
        allp[tnextindex] = px
        timeline.advance()


    zn0 = yspace.function(yspace.interpolate(partial(pde.z_solution, time=T)))
    zn0t = yspace.interpolate(partial(pde.z_t_solution, time=0)) 
    zn1 = yspace.function()
    zn2 = yspace.function()
    qx = pspace.function()
    un0 = yspace.function()
    un1 = yspace.function()

    un0[:] = solver.solve_u(zn0) #积分子
    allu[tnextindex] = un0

    An = solver.A0n()
    # TODO: 未完成 矩阵拼接可以用blockform
    bn = solver.backward_b0(ally[-1], allp[-1])   
    An, bn = solver.backward_boundary(An, bn, isbdof, T-dt)
    xn = solver.mumps_solve(An, bn)

    zn1[:] = xn[:ygodf]
    qx[:] = xn[-pgdof:]
    timeline.backward()

    tnextindex = timeline.current_time_level_index()
    un1[:] = solver.solve_u(zn1)
    allu[tnextindex] = un1

    # 反向求解
    for i in range(nt-1):
        t1 = timeline.prev_time_level()
        tnextindex = timeline.current_time_level_index()-1
        u = yspace.function()    
        b = solver.backward_b(zn0, zn1, ally[tnextindex], allp[tnextindex], t1)
        A,b = solver.forward_boundary(AA, b, isbdof, t1)
        
        x = solver.mumps_solve(A, b)
        
        zn0[:] = zn1
        zn1[:] = x[:ygodf]
        qx[:] = x[-pgdof:]
        u[:] = solver.solve_u(zn1)
        allu[tnextindex] = u
        timeline.backward()


ysolution = yspace.function(yspace.interpolate(partial(pde.y_solution, time=T))) 
usolution = yspace.function(yspace.interpolate(partial(pde.u_solution, time=0))) 
errory = mesh.error(ally[-1], ysolution)
erroru = mesh.error(allu[0], usolution)
print(errory)
print(erroru)
