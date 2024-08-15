from fealpy.experimental import logger
logger.setLevel('WARNING')

from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('pytorch')

from fealpy.experimental.mesh import UniformMesh2D

nelx, nely = 10, 10
domain = [0, 10, 0, 10]
hx = (domain[1] - domain[0]) / nelx
hy = (domain[3] - domain[2]) / nely
mesh = UniformMesh2D(extent=(0, nelx, 0, nely), h=(hx, hy), origin=(domain[0], domain[2]))


















import torch

from fealpy.torch.mesh import QuadrangleMesh

from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mesh = QuadrangleMesh.from_box(box=[0, 32, 0, 20], nx=32, ny=20, device=device)

GD = mesh.geo_dimension()
print("GD:", GD)

NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NC = mesh.number_of_cells()
print("NN:", NN)
print("NE:", NE)
print("NC:", NC)

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
print("node:", node.shape, "\n", node)
print("edge:", edge.shape, "\n", edge)
print("cell:", cell.shape, "\n", cell)

space_u = LagrangeFESpace(mesh, p=1, ctype='C')
print("ldof_u:", space_u.number_of_local_dofs())
print("gdof_u:", space_u.number_of_global_dofs())
uh = space_u.function()
print("uh.shape:", uh.shape, "\n", uh)

tensor_space_u = TensorFunctionSpace(space_u, shape=(2, -1))
print("ldof_tensor_u:", tensor_space_u.number_of_local_dofs())
print("gdof_tensor_u:", tensor_space_u.number_of_global_dofs())
tensor_uh = tensor_space_u.function()
print("tensor_uh.shape:", tensor_uh.shape, "\n", tensor_uh)

space_rho = LagrangeFESpace(mesh, p=1, ctype='D')
print("ldof_rho:", space_rho.number_of_local_dofs())   
print("gdof_rho:", space_rho.number_of_global_dofs())
rhoh = space_rho.function()
print("rhoh.shape:", rhoh.shape, "\n", rhoh)

def SIMP_material_model(rho):
    E0 = 1
    penal = 3
    E = rho ** penal * E0
    return E

def Modified_SIMP_material_model(rho):
    Emin = 1e-3
    E0 = 1
    penal = 3
    E = Emin + rho ** penal * (E0 - Emin)
    return E

 