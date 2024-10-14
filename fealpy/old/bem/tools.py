import numpy as np
from ..mesh import IntervalMesh, TriangleMesh, QuadrangleMesh

def boundary_mesh_build(mesh):
    Boundary_Mesh = {
        "TriangleMesh": IntervalMesh,
        "QuadrangleMesh": IntervalMesh,
        "UniformMesh2d": IntervalMesh,
        "TetrahedronMesh": TriangleMesh,
        "HexahedronMesh": QuadrangleMesh,
        "UniformMesh3d": QuadrangleMesh,
    }
    if type(mesh).__name__ == "UniformMesh3d":
        bd_face = mesh.ds.boundary_face()[:, [0, 2, 3, 1]]
    else:
        bd_face = mesh.ds.boundary_face()
    node = mesh.entity('node')
    old_bd_node_idx = mesh.ds.boundary_node_index()
    new_node = node[old_bd_node_idx]
    aux_idx1 = np.zeros(len(node), dtype=np.int_)
    aux_idx2 = np.arange(len(old_bd_node_idx), dtype=np.int_)
    aux_idx1[old_bd_node_idx] = aux_idx2
    new_cell = aux_idx1[bd_face]
    bd_mesh = Boundary_Mesh[type(mesh).__name__](new_node, new_cell)

    return bd_mesh


def error_calculator(mesh, u, v, q=3, power=2):
    qf = mesh.integrator(q, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    cell = mesh.entity('cell')
    cell_node_location = u[cell]
    if type(mesh).__name__ == "UniformMesh3d":
        bc0 = bcs[0].reshape(-1, 2)  # (NQ0, 2)
        bc1 = bcs[1].reshape(-1, 2)  # (NQ1, 2)
        bc2 = bcs[2].reshape(-1, 2)  # (NQ2, 2)
        bc = np.einsum('im, jn, kl->ijkmnl', bc0, bc1, bc2).reshape(-1, 8)  # (NQ0, NQ1, NQ2, 2, 2, 2)  (NQ0*NQ1*NQ2, 8)

        u = np.einsum('...j, cj->...c', bc, cell_node_location)
    else:
        u = np.einsum('...j, cj->...c', bcs, cell_node_location)
    if callable(v):
        if not hasattr(v, 'coordtype'):
            v = v(ps)
        else:
            if v.coordtype == 'cartesian':
                v = v(ps)
            elif v.coordtype == 'barycentric':
                v = v(bcs)

    if u.shape[-1] == 1:
        u = u[..., 0]

    if v.shape[-1] == 1:
        v = v[..., 0]

    cm = mesh.entity_measure('cell')

    f = np.power(np.abs(u - v), power)

    e = np.einsum('q, qc..., c->c...', ws, f, cm)
    e = np.power(np.sum(e), 1 / power)

    return e