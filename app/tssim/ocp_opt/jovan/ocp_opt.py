from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import TensorFunctionSpace
from ocp_opt_pde import example_1
from solver_update import ocp_opt_solver
from fealpy.fem import DirichletBC  

from functools import partial
from fealpy import logger
from fealpy.solver import spsolve
logger.setLevel('ERROR') #积分子问题

bm.set_backend("numpy")
pde = example_1()
n = 20
q = 3
T = 1
nt = 100
maxit = 10

mesh = TriangleMesh.from_box(pde.domain(), nx=n, ny=n)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

yspace= LagrangeFESpace(mesh, p=1) 
space = LagrangeFESpace(mesh, p=1)
pspace = TensorFunctionSpace(space, (2,-1)) 
solver = ocp_opt_solver(mesh, yspace, pspace, pde, timeline, q=q)

ygdof = yspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

ally = [None]*(nt+1)
allp = [None]*(nt+1)
allu = [None]*(nt+1)
allz = [None]*(nt+1)

y0 = yspace.function(yspace.interpolate(partial(pde.y_solution, time=0)))
y0t = yspace.interpolate(partial(pde.y_t_solution, time=0)) 
p0 = pspace.function(pspace.interpolate(partial(pde.p_solution, time=0))) 
zn = yspace.function(yspace.interpolate(partial(pde.z_solution, time=T)))

q2 = pspace.function()

ally[0] = y0
allp[0] = p0
allz[-1] = zn

A0 = solver.Forward_BForm_A0().assembly()
b0_LForm = solver.Forward_LForm_b0()

FA = solver.Forward_BForm_A().assembly()
Forward_b_LForm = solver.Forward_LForm_b()

An = solver.Forward_BForm_A0().assembly()
bn_LForm = solver.Backward_LForm_bn()

BA = solver.Forward_BForm_A().assembly()
Backward_b_LForm = solver.Backward_LForm_b()

for k in range(maxit):
    y1 = yspace.function()
    p1 = pspace.function()
    
    ## 正向求解第0步
    solver.Forward_0_update(allu[1])
    Fb0 = b0_LForm.assembly()
    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                     gd=(partial(pde.y_solution,time=dt), 
                         partial(pde.p_solution,time=dt)),method='interp')
    A0, b0 = BC.apply(A0, Fb0) 
    x0 = spsolve(A0, b0, solver='mumps')

    y1[:] = x0[:ygdof]
    p1[:] = x0[-pgdof:] 
    ally[1] = y1
    allp[1] = p1
    
    timeline.advance()

    # 正向求解
    for i in bm.arange(2, nt+1):
        t1 = timeline.current_time_level()
        t1index = timeline.current_time_level_index()

        y2 = yspace.function()
        p2 = pspace.function()
        
        solver.Forward_update(ally[t1index-1], ally[t1index], allu[t1index+1], t1+dt)
        Fb = Forward_b_LForm.assembly()
        
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                         gd=(partial(pde.y_solution,time=t1+dt), 
                             partial(pde.p_solution,time=t1+dt)),method='interp')
        Forward_A, Forward_b = BC.apply(FA, Fb)
        Forward_x = spsolve(Forward_A, Forward_b, solver='mumps')
        
        y2[:] = Forward_x[:ygdof]
        p2[:] = Forward_x[-pgdof:]
        ally[t1index+1] = y2
        allp[t1index+1] = p2
        timeline.advance()
    
    ## 反向求解第0步
    t1 = timeline.current_time_level()
    t1index = timeline.current_time_level_index()
    
    zn1 = yspace.function()
    solver.Backward_n_update(ally[t1index-1], allp[t1index-1])
    bn = bn_LForm.assembly() 
     
    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                     gd=(partial(pde.z_solution,time = T-dt), 
                        partial(pde.q_solution,time = T-dt)),method='interp')   
    An, bn = BC.apply(An, bn)
    xn = spsolve(An, bn, solver='mumps')
    zn1[:] = xn[:ygdof]
    q2[:] = xn[-pgdof:]
    allz[t1index-1] = zn1

    timeline.backward()
    
    ## 反向求解 
    for i in bm.arange(nt-1, 0, -1):    
        t1 = timeline.current_time_level()
        t1index = timeline.current_time_level_index()
        zn2 = yspace.function() 

        ##求第i-1步的z,q
        solver.Backward_update(allz[t1index+1], allz[t1index], ally[t1index-1], allp[t1index-1], t1-dt)  
        Bb = Backward_b_LForm.assembly()
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                         gd=(partial(pde.z_solution,time = t1-dt), 
                             partial(pde.q_solution,time = t1-dt)),method='interp') 
        Backward_A, Backward_b = BC.apply(BA, Bb)
        Backward_x = spsolve(Backward_A, Backward_b, solver='mumps')
        
        zn2[:] = Backward_x[:ygdof]
        q2 = Backward_x[-pgdof:]
        allz[t1index-1] = zn2 
        timeline.backward()
    
    z_bar = solver.solve_z_bar(allz)
    un_pre = allu[0]
    if un_pre == None:
        un_pre = 0
    for i in range(nt+1):
       ufunction = yspace.function()
       ufunction[:] = bm.max((0,z_bar)) - allz[i]
       allu[i] = ufunction
    print(f"第{k}次的前后u差别",bm.sum(allu[0] - un_pre))
    
ysolution = yspace.function(yspace.interpolate(partial(pde.y_solution, time=T))) 
psolution = pspace.function(pspace.interpolate(partial(pde.p_solution, time=T))) 
usolution = yspace.function(yspace.interpolate(partial(pde.u_solution, time=0))) 
zsolution = yspace.function(yspace.interpolate(partial(pde.z_solution, time=0))) 
qsolution = pspace.function(pspace.interpolate(partial(pde.q_solution, time=0))) 
errory = mesh.error(ally[-1], ysolution)
errorp = mesh.error(allp[-1], psolution)
erroru = mesh.error(allu[0], usolution)
errorz = mesh.error(allz[0], zsolution)
errorq = mesh.error(q2, qsolution)
print("y误差",errory)
print("p误差",errorp)
print("u误差",erroru)
print("z误差",errorz)
print("q误差",errorq)
