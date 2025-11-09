import json
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("LinearElasticityEigen3d")            # PDE 数据（示例）
mesher = cgraph.create("uniform_tet")                      # 网格生成
spacer = cgraph.create("FunctionSpace")        # 函数空间节点
isDDof = cgraph.create("BoundaryDof")
eig_eq = cgraph.create("LinearElasticityEigenEquation")
eigensolver = cgraph.create("EigenSolver")
dbc = cgraph.create("DirichletBC")

spacer(mesh=mesher(), p=1)

eig_eq(space=spacer(), q=3, material = pde().material, displacement_bc = pde().displacement_bc, is_displacement_boundary = pde().is_displacement_boundary)

eigensolver(
    S=eig_eq().S,
    M=eig_eq().M,
    space=spacer(),

    neigen=6,
    which='SM'
)

# 输出：把网格和第一个模态输出到 WORLD_GRAPH
WORLD_GRAPH.output(mesh=mesher(), uh=eigensolver().modes[0])
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
