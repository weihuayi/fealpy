import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("Timoaxle3d")
mesher = cgraph.create("CreateMesh")
materialer = cgraph.create("TimoaxleMaterial")
spacer = cgraph.create("FunctionSpace")
timoaxle_model = cgraph.create("Timoaxle")

# 连接节点

node = bm.array([[0.0, 0.0, 0.0], [70.5, 0.0, 0.0], [141.0, 0.0, 0.0], [155.0, 0.0, 0.0],
            [169.0, 0.0, 0.0], [213.25, 0.0, 0.0], [257.5, 0.0, 0.0], [301.75, 0.0, 0.0],
            [346.0, 0.0, 0.0], [480.0, 0.0, 0.0], [614.0, 0.0, 0.0], [853.0, 0.0, 0.0],
            [1092.0, 0.0, 0.0], [1334.0, 0.0, 0.0], [1576.0, 0.0, 0.0], [1620.25, 0.0, 0.0],
            [1664.5, 0.0, 0.0], [1708.75, 0.0, 0.0], [1753.0, 0.0, 0.0], [1767.0, 0.0, 0.0],
            [1781.0, 0.0, 0.0], [1851.5, 0.0, 0.0],[1922.0, 0.0, 0.0], [169.0, 0.0, -100.0],
            [213.25, 0.0, -100.0], [257.5, 0.0, -100.0], [301.75, 0.0, -100.0], [346.0, 0.0, -100.0],
            [1576.0, 0.0, -100.0], [1620.25, 0.0, -100.0], [1664.5, 0.0, -100.0], 
            [1708.75, 0.0, -100.0], [1753.0, 0.0, -100.0]], dtype=bm.float64)

cell = bm.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8],[8, 9], [9, 10], [10, 11],
        [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
        [16, 17],[17, 18], [18, 19], [19, 20], [20, 21], 
        [21, 22], [4, 23], [5, 24], [6, 25], [7, 26], [8, 27], 
        [14, 28], [15, 29],[16, 30], [17, 31], [18,32]], dtype=bm.int32)

mesher(node = node, cell = cell)
spacer(mesh=mesher(), p=1)
materialer(beam_E=2.1e11, beam_nu=0.3, axle_E=1.976e6, axle_nu=-0.5)
timoaxle_model(
    space=spacer(),
    beam_material=materialer[0],
    axle_material=materialer[1],
    cell_index=10, 
    external_load=model().external_load,
    dirichlet_idx=model().dirichlet_dof_index, 
    boundary_type=None, 
    load_type=None, 
    load_value=None,
    penalty=1e20
)


# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher(), uh=timoaxle_model())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())