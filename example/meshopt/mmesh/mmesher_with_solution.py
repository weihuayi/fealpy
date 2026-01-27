import argparse

from fealpy.backend import bm
from fealpy.mmesh import MMesher,MeshQuality,Config
from fealpy.mmesh.tool import Poissondata
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
from fealpy.utils import timer
time = timer()
next(time)

parser = argparse.ArgumentParser(description=
        '''
        基于已知函数的移动网格算法测试程序
        ''')
parser.add_argument('--nx',
         default=80, type=int,
         help='初始网格x方向划分单元数')
parser.add_argument('--ny',
         default=80, type=int,
         help='初始网格y方向划分单元数')
parser.add_argument('--case',
            default=4,type=int,
            help='''
            测试用例编号
            1: u = 1/(1+ exp(100*(x+y-1)))
            2: u = 1/2 + 1/2 * tanh(200*(1/12 - (x-0.5)**2 -(y-0.5)**2))
            3: u = tanh( -100 * ( y - 0.5 - 0.25*sin(2*pi*x) ) )
            4: u = tanh(100*(1-x-y)) - tanh(100*(x-y))
            5: u = tanh(50*y) - tanh(50*(x-y-0.5))
            6: 五个小圆形初值
                ''')
parser.add_argument('--tau',
        default=0.004, type=float,
        help='时间步长控制参数tau')
parser.add_argument('--gamma',
        default=1.25, type=float,
        help='指数控制参数gamma')
parser.add_argument('--mol_times',
        default=6, type=int,
        help='磨光次数')
parser.add_argument('--mmesher',
        default='MetricTensorAdaptive', type=str,
        help='移动网格算法名称')
parser.add_argument('--method',
        default='BDF_LFP', type=str,
        help='ODE求解方法')
parser.add_argument('--monitor',
        default='linear_int_error', type=str,
        help='监控函数类型')
parser.add_argument('--mol_method',
        default='huangs_method', type=str,
        help='磨光方法')
parser.add_argument("--plot_timemesh",
        default=False, type=bool,
        help='是否输出时间网格的动态图')
parser.add_argument('--is_return_info',
        default=False, type=bool,
        help='是否返回移动网格信息')
parser.add_argument('--output_vtu',
        default=False, type=bool,
        help='是否输出.vtu可视化文件')

args = parser.parse_args()
options = vars(args)

def moving_mesher_with_solution(nx = 10 , ny = 10 , 
                                case:int = 1 ,
                                tau:float = 0.004 ,
                                gamma:float = 1.0,
                                mol_times:int = 4,
                                mmesher = "MetricTensorAdaptive",
                                method:str = 'BDF_LBFGS',
                                monitor:str = 'arc_length',
                                mol_method:str = 'projector',
                                is_return_info = False,
                                plot_timemesh = False,
                                output_vtu = False):

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
      u = 'tanh(100*(1-x-y)) - tanh(100*(x-y))'
   elif case == 5:
      u = 'tanh(50*y) - tanh(50*(x-y-0.5))'
   elif case == 6:
      X = '-2 + 4*x'
      Y = '-2 + 4*y'
      R = 50
      u = f'tanh({R}*(({X})**2 + ({Y})**2 -1/8))\
          + tanh({R}*(({X}-0.5)**2 + ({Y}-0.5)**2 -1/8))\
          + tanh({R}*(({X}-0.5)**2 + ({Y}+0.5)**2 -1/8))\
          + tanh({R}*(({X}+0.5)**2 + ({Y}-0.5)**2 -1/8))\
          + tanh({R}*(({X}+0.5)**2 + ({Y}+0.5)**2 -1/8))'

      
   var_list = ['x','y']
   mesh.meshdata['vertices'] = vertices
   pde = Poissondata(u,var_list)
   beta = 0.5
   config = Config()
   config.t_max = 0.25
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
   mm.set_monitor(monitor)
   mm.set_mol_method(mol_method)

   # mesh,uh = mm.run()
   adaptiver = mm.instance
   time.send(f"Initialized the moving mesh adaptiver: {mmesher}")
   if mmesher in ["MetricTensorAdaptive","EAGAdaptiveHuang","EAGAdaptiveFB",
                  "EAGAdaptiveXHuang","EAGAdaptiveXFB","MetricTensorAdaptiveX"]:
      ret_info = adaptiver.mesh_redistributor(return_info = is_return_info ,
                                             method=method,
                                             return_timemesh=plot_timemesh)
   else:
      mesh,uh = adaptiver.mesh_redistributor()
   time.send(f"Completed the moving mesh adaptiver: {mmesher}")
   next(time)
   uh = space.interpolate(pde.solution)
   error1 = mesh.error(uh , pde.solution)
   print("=== moving mesh ===")
   print("Number of cells: ", mesh.number_of_cells())
   print("before move:",error0)
   print("after move:",error1)
   
   if mmesher in ["MetricTensorAdaptive","EAGAdaptiveHuang","EAGAdaptiveFB",
                  "EAGAdaptiveXHuang","EAGAdaptiveXFB","MetricTensorAdaptiveX"]:
      if is_return_info:
         info = ret_info["info"]
         I_h = info[0]
         I_t= info[1]
         cm_min = info[2]
         steps = len(I_h)

         # 科研风子图
         plt.style.use('seaborn-v0_8-whitegrid')
         plt.rcParams.update({
             'font.family': 'sans-serif',
             'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
             'axes.titlesize': 11,
             'axes.labelsize': 10,
             'xtick.labelsize': 9,
             'ytick.labelsize': 9,
             'legend.fontsize': 9
         })
         # fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
         plots = [
             (range(steps), I_h, 'Functional $I_h$', 'Step', 'Value'),
             (range(steps-1), I_t, r'$\Delta I_h$ per step', 'Step', 'Value'),
             (range(steps), cm_min, 'Minimum Cell Volume', 'Step', 'Volume')
         ]
         for xs, ys, title, xl, yl in plots:
             fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200,constrained_layout=True)
             ax.plot(xs, ys, '-o', ms=4, lw=1.2, color='#1f77b4',
                     markerfacecolor='white', markeredgewidth=0.9)
             ax.set_title(title, fontweight='bold')
             ax.set_xlabel(xl)
             ax.set_ylabel(yl)
             ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
             ax.spines['top'].set_visible(False)
             ax.spines['right'].set_visible(False)
      if plot_timemesh:
         time_mesh = ret_info["time_mesh"]
         
      logic_mesh = adaptiver.logic_mesh
      M = adaptiver.M
      mesh.node = ret_info["X"]
      mq = MeshQuality(mesh, logic_mesh, M)
      
      print("Q_eq:", mq.Q_eq())
      print("Q_ali:", mq.Q_ali())
      print("Q_geo:", mq.Q_geo())
      mq.stats_hist_metrics()
      
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
   else:
      pass
   fig = plt.figure(dpi=200)
   ax = fig.add_subplot(111)
   mesh.add_plot(ax, linewidths=0.25)
   plt.show()
   
   if output_vtu:
      mesh.to_vtk(f'{mmesher}_with_solution_case{case}_nx{nx}_ny{ny}.vtu')


moving_mesher_with_solution(**options)