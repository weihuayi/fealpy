
from mpi4py import MPI
from fealpy.backend import backend_manager as bm
from fealpy.mesh import Mesh, QuadrangleMesh
from fealpy.mesh.parallel import ParallelMesh, split_homogeneous_mesh

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

DEVICE = 'cpu'
bm.set_backend('numpy')

# Construct and split the mesh in process 0
# 准备工作：在第 0 进程中创建并分割网格

if RANK == 0:
    full_mesh = QuadrangleMesh.from_box([-1, 1, -1, 1], nx=20, ny=20, device=DEVICE)
    TOTAL_NC = full_mesh.number_of_cells()
    ranges = []
    start = 0
    step = TOTAL_NC // SIZE + 1

    for i in range(SIZE):
        stop = start + step
        if stop >= TOTAL_NC:
            stop = TOTAL_NC
        ranges.append(bm.arange(start, stop, 1, dtype=full_mesh.itype))
        start = stop

    data_list = list(split_homogeneous_mesh(full_mesh, masks=ranges))
else:
    data_list = [None,] * SIZE

data = COMM.scatter(data_list, root=0)
del data_list

# Make submesh for each process
# 在每个进程中建立子网格

pmesh = ParallelMesh(RANK, *data)

# Check some info
print(f"[{RANK}] Total number of nodes 总顶点数（整个网格）", pmesh.Count_all('node'))
print(f"[{RANK}] Number of nodes 顶点数（包括虚点）", pmesh.count('node'))
print(f"[{RANK}] Number of real nodes 真实顶点数", pmesh.count_real('node'))


##################################################
# Example: Integral on cells, then sum to nodes
# 例子：在子网格的单元内积分，然后求和到顶点
##################################################

# 1. 在每个子网格中积分并求和到顶点

def func_example(p):
    PI = bm.pi
    x, y = p[..., 0], p[..., 1]
    return bm.sin(PI*x) * bm.sin(PI*y)

def integral_on_mesh_cells_to_nodes(mesh: Mesh, func):
    qf = mesh.quadrature_formula(3, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    qpoints = mesh.bc_to_point(bcs)
    val = func(qpoints)
    ms = mesh.entity_measure('cell')
    integral_on_cell = bm.einsum('q, cq, c -> c', ws, val, ms)

    NN = mesh.number_of_nodes()
    integral_on_node = bm.zeros((NN, ), dtype=mesh.ftype, device=DEVICE)
    NVC = mesh.number_of_vertices_of_cells()[0]

    for local_node in range(NVC):
        integral_on_node = bm.index_add(
            integral_on_node, mesh.cell[:, local_node], integral_on_cell
        )

    return integral_on_node

integral_on_node = integral_on_mesh_cells_to_nodes(pmesh, func_example)

# 2. 获得每个子网格边界顶点上的数据，进程间通信

integral_on_bd_node = integral_on_node[pmesh.boundary_node_flag()]
recv_list = pmesh.Converge('node', integral_on_bd_node)

# 3. 每个进程叠加各自收到的信息，完成计算
# （结果仍存在于各个进程中，可进行后续计算，若有）

for i in range(SIZE):
    if i == RANK: # no data for self
        continue
    integral_recv, local_index = recv_list[i]
    integral_on_node = bm.index_add(integral_on_node, local_index, integral_recv)

# 4. 检验结果：拼起来检查是否和单进程计算结果相同

feal_node_flag = pmesh.real_flag('node')
integral_on_real_node = integral_on_node[feal_node_flag]
int_on_real_node_collected = COMM.gather(
    (integral_on_real_node, pmesh.global_indices('node')[feal_node_flag]),
    root=0
)

if RANK == 0:
    TOTAL_NN = full_mesh.number_of_nodes()
    integral_on_all_nodes = bm.zeros((TOTAL_NN, ), dtype=full_mesh.ftype, device=DEVICE)
    for data, indices in int_on_real_node_collected:
        integral_on_all_nodes = bm.set_at(
            integral_on_all_nodes,
            indices,
            data
        )

    integral_on_nodes_expected = integral_on_mesh_cells_to_nodes(full_mesh, func_example)
    print("Test passed: ", bm.allclose(integral_on_all_nodes, integral_on_nodes_expected))
