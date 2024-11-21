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
zn = yspace.function(yspace.interpolate(partial(pde.z_solution, time=T)))
znt = yspace.interpolate(partial(pde.z_t_solution, time=0)) 

zn0 = yspace.function()
zn0[:] = zn[:]
zn1 = yspace.function()
zn2 = yspace.function()
q2 = pspace.function()

un = yspace.function()
un[:] = solver.z_to_u(zn) #积分子
allu[-1] = un
ally[0] = y0
allp[0] = p0

A0 = solver.Forward_BForm_A0().assembly()
b0_LForm = solver.Forward_LForm_b0()

A = solver.Forward_BForm_A().assembly()
Forward_b_LForm = solver.Forward_LForm_b()

An = solver.Forward_BForm_A0().assembly()
bn_LForm = solver.Backward_LForm_bn()

Backward_A = solver.Forward_BForm_A().assembly()
Backward_b_LForm = solver.Backward_LForm_b()

for k in range(maxit):
    y1 = yspace.function()
    p1 = pspace.function()

    solver.Forward_0_update(allu[1])
    b0 = b0_LForm.assembly()

    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                     gd=(partial(pde.y_solution,time=dt), 
                         partial(pde.p_solution,time=dt)),method='interp')
    A0, b0 = BC.apply(A0, b0) 
    x0 = spsolve(A0, b0, solver='scipy')

    y1[:] = x0[:ygodf]
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
        Forward_b = Forward_b_LForm.assembly()
        
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                         gd=(partial(pde.y_solution,time=t1+dt), 
                             partial(pde.p_solution,time=t1+dt)),method='interp')
        Forward_A, Forward_b = BC.apply(A, Forward_b)
        Forward_x = spsolve(Forward_A, Forward_b, solver='scipy')
        
        y2[:] = Forward_x[:ygodf]
        p2[:] = Forward_x[-pgdof:]
        ally[t1index+1] = y2
        allp[t1index+1] = p2
        timeline.advance()
    
    ## 反向求解第0步
    t1 = timeline.current_time_level()
    t1index = timeline.current_time_level_index()
    
    solver.Backward_n_update(ally[t1index-1], allp[t1index-1])
    bn = bn_LForm.assembly() 
     
    BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                     gd=(partial(pde.y_solution,time = T-dt), 
                        partial(pde.p_solution,time = T-dt)),method='interp')   
    An, bn = BC.apply(An, bn)
    xn = spsolve(An, bn, solver='scipy')
    zn1[:] = xn[:ygodf]
    q2[:] = xn[-pgdof:]
    
    un1 = yspace.function()  
    un1[:] = solver.z_to_u(zn1)
    allu[t1index-1] = un1 
    timeline.backward()
    
    ## 反向求解 
    for i in bm.arange(nt-1, 0, -1):    
        t1 = timeline.current_time_level()
        t1index = timeline.current_time_level_index()
        
        
        ##求第i-1步的z,q
        solver.Backward_update(zn0, zn1, ally[t1index-1], allp[t1index-1], t1-dt)  
        Backward_b = Backward_b_LForm.assembly()
        BC = DirichletBC(space=(yspace,pspace), threshold=(None, None), 
                         gd=(partial(pde.y_solution,time = t1-dt), 
                             partial(pde.p_solution,time = t1-dt)),method='interp') 
        Backward_A, Backward_b = BC.apply(Backward_A, Backward_b)
        Backward_x = spsolve(Backward_A, Backward_b, solver='scipy')
        
        zn2[:] = Backward_x[:ygodf]
        q2 = Backward_x[-pgdof:]
        
        ##求第i-1步的u
        un2 = yspace.function()  
        un2[:] = solver.z_to_u(zn2)
        allu[t1index-1] = un2
        
        zn0[:] = zn1[:]
        zn1[:] = zn2[:]
        timeline.backward()
    
ysolution = yspace.function(yspace.interpolate(partial(pde.y_solution, time=T))) 
usolution = yspace.function(yspace.interpolate(partial(pde.u_solution, time=0))) 
errory = mesh.error(ally[-1], ysolution)
erroru = mesh.error(allu[0], usolution)
print(errory)
print(erroru)
