import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

uav = cgraph.create("UAVPathPlanning")

uav(
    start_x = 900,
    start_y = 100,
    end_x = 100,
    end_y = 800,
    opt_alg = "AnimatedOatOpt"
)

WORLD_GRAPH.output(
    PATH=uav().PATH,
    Distance=uav().Distance
)

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())