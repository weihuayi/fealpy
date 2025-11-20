import fealpy.cgraph as cgraph
WORLD_GRAPH = cgraph.WORLD_GRAPH
sample = cgraph.create("Sample")
sample(domain=[-1,2,0,3], mode="linspace", n=4, boundary=True)
# sample(domain=[-1,2,0,3], mode="random", n=2, boundary=True)

WORLD_GRAPH.output(sa=sample())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

