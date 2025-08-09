
import fealpy.cgraph as cgraph
WORLD_GRAPH = cgraph.WORLD_GRAPH
pde = cgraph.create("StationaryNS2d")
mesher = cgraph.create("Box2d")
equation = cgraph.create("StationaryNSEquation")
simulation = cgraph.create("StationaryNSSimulation")
StationaryNSRun = cgraph.create("StationaryNSRun")
equation(pde = pde())

simulation(equation=equation()) 
StationaryNSRun(
    maxstep=1000,
    simulation = simulation(),
    tol=1e-6,
    pde=pde()
)
WORLD_GRAPH.output_node(
    uh1 = StationaryNSRun().uh1,  
    ph1 = StationaryNSRun().ph1
)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())