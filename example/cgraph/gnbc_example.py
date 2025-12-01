import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
WORLD_GRAPH = cgraph.WORLD_GRAPH
pde = cgraph.create("CouetteFlow")
mesher = cgraph.create("Box2d")
phispace = cgraph.create("FunctionSpace")   
pspace = cgraph.create("P0FunctionSpace")
space = cgraph.create("FunctionSpace")
uspace = cgraph.create("TensorFunctionSpace")
simulation = cgraph.create("GNBCSimulation")

pde(
    eps = 1e-10,
    T = 2,
    h = 1/256,
    theta = 77.6
)
mesher(
    domain = pde().domain,
    nx = pde().nx,
    ny = pde().ny
)
phispace(mesh = mesher(), p=1)
pspace(mesh = mesher(), ctype='D')
space(mesh = mesher(), p=2)
uspace(mesh = mesher(), p=2, gd = 2)

simulation(
    param_list = pde().param_list,
    init_phi = pde().init_phi,
    is_uy_Dirichlet = pde().is_uy_Dirichlet,
    is_up_boundary = pde().is_up_boundary,
    is_down_boundary = pde().is_down_boundary,
    is_wall_boundary = pde().is_wall_boundary,
    u_w = pde().u_w,
    mesh = mesher(),
    nt = pde().nt,
    phispace = phispace(),
    space = space(),
    pspace = pspace(),
    uspace = uspace(),
    output_dir = "/home/edwin/output", 
    q = 5)

WORLD_GRAPH.output(max_u_up=simulation().max_u_up, 
                   min_u_up=simulation().min_u_up,
                   max_u_down=simulation().max_u_down,
                   min_u_down=simulation().min_u_down)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())