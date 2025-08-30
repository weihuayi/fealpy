import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

cec_2022 = cgraph.create("CEC2022")

cec_2022(
    func_num=1, 
    dim=2, 
    opt_alg="ParticleSwarmOpt",
    NP=100,
    MaxIT=1000
) 

WORLD_GRAPH.output_node(
    Optimal_sol=cec_2022().Optimal_sol, 
    Optimal_val=cec_2022().Optimal_val
)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())