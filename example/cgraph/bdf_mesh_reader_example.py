import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("BdfMeshReader")

# 连接节点
mesh=mesher(input_bdf_file="./test_bdf_parser.bdf")



# 最终连接到图输出节点上
WORLD_GRAPH.output_node(mesh=mesh)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())