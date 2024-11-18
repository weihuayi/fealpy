from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import TensorFunctionSpace
from ocp_opt_pde import example_1
#from solver import ocp_opt_solver
from solver_update import ocp_opt_solver
from fealpy.fem import DirichletBC  

from scipy.sparse import coo_array, bmat
from functools import partial
from fealpy import logger
from fealpy.solver import spsolve
logger.setLevel('ERROR') #积分子问题

bm.set_backend("numpy")
pde = example_1()
n = 20
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
    #A0 = solver.A0n()
    A0 = solver.FBForm_A0().assembly()
    b0 = solver.FLform_b0(allu[1]).assembly()
   
    #A0, b0 = solver.forward_boundary(A0, b0, isbdof, dt)
    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), gd=(partial(pde.y_solution,time=0), partial(pde.p_solution,time = 0)),method='interp')
    A0, b0 = BC.apply(A0, b0) #检查一下边界条件的处理   ，这里的A矩阵不一样。
    x0 = spsolve(A0, b0, solver='scipy')

    y1 = yspace.function()
    p1 = pspace.function()

    y1[:] = x0[:ygodf]
    p1[:] = x0[-pgdof:] #p1x1, p1x2
    ally[1] = y1
    allp[1] = p1
    timeline.advance()

    A_Bform= solver.FBform_A()
    b_lform = solver.FLform_b()
    AA = A_Bform.assembly()
    # 正向求解
    for i in range(nt-1):
        t1 = timeline.next_time_level()
        tnextindex = timeline.current_time_level_index()+1 

        y2 = yspace.function()
        px = pspace.function()
        solver.FLformb_update(y0, y1, allu[tnextindex], t1)
        b = b_lform.assembly()
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), gd=(partial(pde.y_solution,time=t1), partial(pde.p_solution,time = t1)),method='interp')
        A, b = BC.apply(AA, b)
        x = spsolve(A, b, solver='scipy')
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
    An = solver.FBForm_A0().assembly()
    bn = solver.BLform_b0(ally[-1], allp[-1]).assembly()
    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), gd=(partial(pde.y_solution,time=T), partial(pde.p_solution,time = T)),method='interp')   
    An, bn = BC.apply(An, bn)
    xn = solver.mumps_solve(An.tocoo(), bn)

    zn1[:] = xn[:ygodf]
    qx[:] = xn[-pgdof:]
    timeline.backward()

    tnextindex = timeline.current_time_level_index()
    un1[:] = solver.solve_u(zn1)
    allu[tnextindex] = un1

    A_Bform= solver.FBform_A()
    b_lform = solver.BLform_b()
    A = A_Bform.assembly()

    # 反向求解
    for i in range(nt-1):
        t1 = timeline.prev_time_level()
        tnextindex = timeline.current_time_level_index()-1
        u = yspace.function()  
        solver.BLformb_update(zn0, zn1, ally[tnextindex], allp[tnextindex], t1)  
        b = b_lform.assembly()
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), gd=(partial(pde.y_solution,time=t1), partial(pde.p_solution,time = t1)),method='interp') 
        A, b = BC.apply(A, b)
        x = solver.mumps_solve(A.tocoo(), b)
        
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
