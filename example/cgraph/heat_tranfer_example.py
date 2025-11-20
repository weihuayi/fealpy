import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
WORLD_GRAPH = cgraph.WORLD_GRAPH
generation = cgraph.create("HeatTransferParticleGeneration")
iterative = cgraph.create("HeatTransferParticleIterativeUpdate")

generation(
    dx=0.02,
    dy=0.02,
)

iterative(mesh = generation().mesh, maxstep=1000, dx=0.02, kernel="quintic", 
        dt=0.00045454545454545455,output_dir="/home/output")
WORLD_GRAPH.output(velocity=iterative().velocity, 
                   pressure=iterative().pressure,
                   temperature=iterative().temperature)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())