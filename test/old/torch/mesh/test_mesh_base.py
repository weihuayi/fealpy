

from fealpy.torch.mesh.mesh_base import MeshDS, entitymethod


def test_entity():
    mesh = MeshDS(2)

    mesh.cell = object()
    mesh.face = object()
    mesh.edge = object()
    mesh.node = object()

    assert len(mesh._entity_storage) == 3
    assert mesh.cell is mesh._entity_storage[2]
    assert mesh.face is mesh._entity_storage[1]
    assert mesh.edge is mesh._entity_storage[1]
    assert mesh.node is mesh._entity_storage[0]


class ExampleMeshDS(MeshDS):

    @entitymethod(0)
    def make_node(self):
        return "this is node"

    @entitymethod(1)
    def generate_edge(self):
        return "this is custom edge"

    @entitymethod(top_dim=4)
    def get_cell(self):
        return "this is cell in higher dimension"


def test_entity_factory():
    mesh = ExampleMeshDS(4)

    assert mesh.node == "this is node"
    assert mesh.edge == "this is custom edge"
    assert mesh.cell == "this is cell in higher dimension"
    assert mesh.face is None

    mesh.edge = "changed edge"
    assert mesh.entity(1) == "changed edge"

    del mesh.edge
    assert mesh.entity(1) == "this is custom edge"
