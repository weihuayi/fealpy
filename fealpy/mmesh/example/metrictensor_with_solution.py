from fealpy.backend import bm
from fealpy.mmesh import (MMesher,MetricTensorAdaptive
                         ,MetricTensorAdaptiveX,MeshQuality)
from fealpy.functionspace import LagrangeFESpace
from fealpy.mmesh.tool import Poissondata
from fealpy.mesh import TriangleMesh
from fealpy.mmesh import Config
import matplotlib.pyplot as plt


def moving_mesher_with_solution(nx = 10 , ny = 10 , 
                                case:int = 1 ,
                                tau:float = 0.004 ,
                                gamma:float = 1.0,
                                mol_times:int = 4,
                                mmesher = "MetricTensorAdaptive",
                                method:str = 'BDF_LBFGS',
                                return_info = False,
                                plot_timemesh = False):

   box = [0,1,0,1]
   mesh = TriangleMesh.from_box_cross_mesh(box, nx=nx, ny=ny)
   vertices = bm.array([[0,0],[1,0],[1,1],[0,1]],dtype = bm.float64)
   
   if case == 1:
      u = '1/(1+ exp(100*(x+y-1)))'
   elif case == 2:
      u = '1/2 + 1/2 * tanh(200*(1/12 - (x-0.5)**2 -(y-0.5)**2))'
   elif case == 3:
      u = 'tanh( -100 * ( y - 0.5 - 0.25*sin(2*pi*x) ) )'
   elif case == 4:
      u = 'tanh(50*(2-2*y-2*x)) - tanh(50*(2*x-2*y))'
   elif case == 5:
      u = 'tanh(50*y) - tanh(50*(x-y-0.5))'
      
   var_list = ['x','y']
   mesh.meshdata['vertices'] = vertices
   pde = Poissondata(u,var_list)
   beta = 1
   config = Config()
   config.pde = pde
   config.active_method = mmesher
   config.mol_times = mol_times
   config.is_pre = False
   config.tau = tau
   config.gamma = gamma
   
   p = 1
   space = LagrangeFESpace(mesh, p=p)
   uh = space.interpolate(pde.solution)
   error0 = mesh.error(uh, pde.solution)

   mm = MMesher(mesh=mesh,uh=uh, space=space,beta=beta, config=config)
   mm.initialize()
   mm.set_interpolation_method('solution')
   mm.set_monitor('linear_int_error')
   mm.set_mol_method('huangs_method')

   # mesh,uh = mm.run()
   MetricTensor_adaptive = mm.instance
   MetricTensor_adaptive.monitor()
   MetricTensor_adaptive.mol_method()

   ret_info = MetricTensor_adaptive.mesh_redistributor(return_info= return_info ,
                                                       method=method,
                                                       return_timemesh=plot_timemesh)
   if return_info:
      info = ret_info["info"]
   if plot_timemesh:
      time_mesh = ret_info["time_mesh"]
      
   logic_mesh = MetricTensor_adaptive.logic_mesh
   M = MetricTensor_adaptive.M
   mesh.node = ret_info["X"]
   uh = space.interpolate(pde.solution)
   mq = MeshQuality(mesh, logic_mesh, M)

   error1 = mesh.error(uh , pde.solution)
   print("=== MetricTensorAdaptive moving mesh ===")
   print("Number of cells: ", mesh.number_of_cells())
   print("before move:",error0)
   print("after move:",error1)
   print("Q_eq:", mq.Q_eq())
   print("Q_ali:", mq.Q_ali())
   print("Q_geo:", mq.Q_geo())
   mq.stats_hist_metrics()
   fig = plt.figure(dpi=100)
   ax = fig.add_subplot(111)
   mesh.add_plot(ax)

   
   I_h = info[0]
   cm_min = info[1]
   steps = len(I_h)
   plt.figure()
   plt.plot(range(steps), I_h, '-o')
   plt.xlabel('Step')
   plt.ylabel('Functional I_h')
   plt.title('Functional I_h vs Step')
   plt.grid()

   plt.figure()
   plt.plot(range(steps), cm_min, '-o')
   plt.xlabel('Step')
   plt.ylabel('Minimum Cell Volume')
   plt.title('Minimum Cell Volume vs Step')
   plt.grid()
   plt.show()
   
   if plot_timemesh:
      from fealpy.mmesh.tool import AnimationTool  
      fig = plt.figure(dpi=200)
      fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
      ax = fig.add_subplot(111)
      def update_mesh(frame):
         X = time_mesh[frame]
         ax.clear()
         ax.set_aspect('equal')
         mesh.node = X
         mesh.add_plot(ax,linewidths=0.3)
         return ax
      save_path = f"{mmesher}_moving_mesh_case{case}_nx{nx}_ny{ny}.gif"
      anim_tool = AnimationTool(update_mesh,fps= 3, frames=len(time_mesh),
                                save_path=save_path, fig=fig)
      anim_tool.run()
    
n = 80
moving_mesher_with_solution(nx = n , ny = n , case=5,
                                tau=0.005,
                                gamma=1,
                                mol_times=8,
                                method = 'BDF_LBFGS',
                                mmesher= "MetricTensorAdaptive",
                                return_info= True,
                                plot_timemesh = True)