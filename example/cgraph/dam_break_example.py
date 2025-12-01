import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH
generation = cgraph.create("DamBreakParticleGeneration")
iterative = cgraph.create("DamBreakParticleIterativeUpdate")

generation(
    dx=0.03,
    dy=0.03,
)
iterative(maxstep=2000, dt=0.001,rho0=1000,pp=generation().pp,
    bpp=generation().bpp,output_dir="/home/edwin/output")
WORLD_GRAPH.output(velocity=iterative().velocity, pressure=iterative().pressure)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
