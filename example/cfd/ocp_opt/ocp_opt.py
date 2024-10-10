#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ocp-opt.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Sep 2024 04:50:58 PM CST
	@bref 
	@ref 
'''  
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.timeintegratoralg import UniformTimeLine

from ocp_opt_pde import example_1
from solver import ocp_opt_solver

from scipy.sparse import coo_array, bmat
from functools import partial
from fealpy.experimental import logger
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
pspace= LagrangeFESpace(mesh, p=1) 
solver = ocp_opt_solver(mesh, yspace, pspace, pde, timeline)

ygodf = yspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
yisbdof = yspace.is_boundary_dof()
pisbdof = pspace.is_boundary_dof()
isbdof = bm.concatenate([yisbdof, pisbdof, pisbdof], axis=0)

ally = [None]*(nt+1)
allpx1 = [None]*(nt+1)
allpx2 = [None]*(nt+1)
allu = [None]*(nt+1)

y0 = yspace.function(yspace.interpolate(partial(pde.y_solution, time=0)))
y0t = yspace.interpolate(partial(pde.y_t_solution, time=0)) 
p0x1 = pspace.function(pspace.interpolate(partial(pde.px1_solution, time=0)))
p0x2 = pspace.function(pspace.interpolate(partial(pde.px2_solution, time=0)))
ally[0] = y0
allpx1[0] = p0x1
allpx2[0] = p0x2

for k in range(maxit):
    A0 = solver.A0n()
    b0 = solver.forward_b0(allu[1])
    A0, b0 = solver.forward_boundary(A0, b0, isbdof, dt)
    x0 = solver.mumps_solve(A0, b0)

    y1 = yspace.function()
    p1x1 = pspace.function()
    p1x2 = pspace.function()
    y1[:] = x0[:ygodf]
    p1x1[:] = x0[ygodf:-pgdof]
    p1x2[:] = x0[-pgdof:]
    ally[1] = y1
    allpx1[1] = p1x1
    allpx2[1] = p1x2
    timeline.advance()

    AA = solver.A()
    #正向求解
    for i in range(nt-1):
        t1 = timeline.next_time_level()
        tnextindex = timeline.current_time_level_index()+1

        y2 = yspace.function()
        px1 = pspace.function()
        px2 = pspace.function()
        b = solver.forward_b(y0, y1, allu[tnextindex], t1)
        A,b = solver.forward_boundary(AA, b, isbdof, t1)
        
        x = solver.mumps_solve(A, b)
        y1[:] = y0
        y2[:] = x[:ygodf]
        px1[:] = x[ygodf:-pgdof]
        px2[:] = x[-pgdof:]
        ally[tnextindex] = y2
        allpx1[tnextindex] = px1
        allpx2[tnextindex] = px2
        timeline.advance()


    zn0 = yspace.function(yspace.interpolate(partial(pde.z_solution, time=T)))
    zn0t = yspace.interpolate(partial(pde.z_t_solution, time=0)) 
    zn1 = yspace.function()
    zn2 = yspace.function()
    qx1 = pspace.function()
    qx2 = pspace.function()
    un0 = yspace.function()
    un1 = yspace.function()

    un0[:] = solver.solve_u(zn0) #积分子
    allu[tnextindex] = un0

    An = solver.A0n()
    bn = solver.backward_b0(ally[-1], allpx1[-1], allpx2[-1])     
    An, bn = solver.backward_boundary(An, bn, isbdof, T-dt)
    xn = solver.mumps_solve(An, bn)

    zn1[:] = xn[:ygodf]
    qx1[:] = xn[ygodf:-pgdof]
    qx2[:] = xn[-pgdof:]
    timeline.backward()

    tnextindex = timeline.current_time_level_index()
    un1[:] = solver.solve_u(zn1)
    allu[tnextindex] = un1

    # 反向求解
    for i in range(nt-1):
        t1 = timeline.prev_time_level()
        tnextindex = timeline.current_time_level_index()-1
        u = yspace.function()    

        b = solver.backward_b(zn0, zn1, ally[tnextindex], allpx1[tnextindex], allpx2[tnextindex], t1)
        A,b = solver.forward_boundary(AA, b, isbdof, t1)
        
        x = solver.mumps_solve(A, b)
        
        zn1[:] = zn0
        zn2[:] = x[:ygodf]
        qx1[:] = x[ygodf:-pgdof]
        qx2[:] = x[-pgdof:]
        u[:] = solver.solve_u(zn2)
        allu[tnextindex] = u
        timeline.backward()


ysolution = yspace.function(yspace.interpolate(partial(pde.y_solution, time=T))) 
usolution = yspace.function(yspace.interpolate(partial(pde.u_solution, time=0))) 
errory = mesh.error(ally[-1], ysolution)
erroru = mesh.error(allu[0], usolution)
print(errory)
print(erroru)
