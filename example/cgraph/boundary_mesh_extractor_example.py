import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher_origin = cgraph.create("Box2d")
mesher = cgraph.create("BoundaryMeshExtractor")

# 连接节点
mesher(input_mesh=mesher_origin())


# 最终连接到图输出节点上
WORLD_GRAPH.output_node(boundary_mesh=mesher().boundary_mesh,
                        node_idx=mesher().face_idx, face_idx=mesher().face_idx)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())