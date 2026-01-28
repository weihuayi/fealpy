import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
WORLD_GRAPH = cgraph.WORLD_GRAPH
pde = cgraph.create("CouetteFlow")
phispace = cgraph.create("FunctionSpace")   
pspace = cgraph.create("P0FunctionSpace")
space = cgraph.create("FunctionSpace")
uspace = cgraph.create("TensorFunctionSpace")
solver = cgraph.create("GNBCSolver")

pde(
    eps=1e-10, 
    h=1/256,
)

phispace(mesh = pde().mesh, p=1)
pspace(mesh = pde().mesh, ctype='D')
space(mesh = pde().mesh, p=2)
uspace(mesh = pde().mesh, p=2, gd = 2)

solver(R = pde().R,
       L_s = pde().L_s,
       epsilon = pde().epsilon,
       L_d = pde().L_d,
       lam = pde().lam,
       V_s = pde().V_s,
       s = pde().s,
       theta_s = pde().theta_s,
       is_wall_boundary = pde().is_wall_boundary,
       is_up_boundary = pde().is_up_boundary,
       is_down_boundary = pde().is_down_boundary,
       is_uy_Dirichlet = pde().is_uy_Dirichlet,
       u_w = pde().u_w,
       mesh = pde().mesh,
       init_phi = pde().init_phi,
       phispace = phispace(),
       pspace = pspace(),
       space = space(),
       uspace = uspace(),
       h = pde().h,
       output_dir = "/home/output", 
       q = 5)

WORLD_GRAPH.output(max_u_up=solver().max_u_up, 
                   min_u_up=solver().min_u_up,
                   max_u_down=solver().max_u_down,
                   min_u_down=solver().min_u_down)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())