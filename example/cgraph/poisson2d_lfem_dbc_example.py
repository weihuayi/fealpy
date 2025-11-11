from fealpy import logger
import fealpy.cgraph as cgraph

logger.handlers.clear()
logger.setLevel('INFO')

pde = cgraph.create("Poisson2d")
mesher = cgraph.create("Box2d")
spacer = cgraph.create("FunctionSpace")
poisson_eq = cgraph.create("PoissonEquationDBC")
solver = cgraph.create("CGSolver")
mesh2d3d = cgraph.create("MeshDimensionUpgrading")

pde(example=3)

mesher(
    domain=[-1, 1, -1, 1],
    nx=20,
    ny=20
)
spacer(mesh=mesher(), p=1)
poisson_eq(
    space=spacer(),
    q=3,
    source=pde().source,
    gd=pde().dirichlet,
)
solver(
    A=poisson_eq().operator,
    b=poisson_eq().source
)
mesh2d3d(mesh=mesher(), z=solver())

original_graph = cgraph.WORLD_GRAPH
original_graph.output(mesh=mesh2d3d(), uh=solver())
original_graph.register_error_hook(print)

# Here we test the serialization ability (optional).
# 1) dump into json
data = cgraph.dump(original_graph)
# 2) load from json
recovered_graph = cgraph.load(data)

# Let FEALPy's logger output to the graph
logger.addHandler(recovered_graph.new_log_handler())
# Print when an error occurs
recovered_graph.register_error_hook(print)
# Run
recovered_graph.execute()
# Get the result
result = recovered_graph.get()
# Print all log messages
print(*recovered_graph.log_message(), sep="\n")

mesh = result["mesh"]

from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(projection="3d")
mesh.add_plot(axes)

plt.show()
