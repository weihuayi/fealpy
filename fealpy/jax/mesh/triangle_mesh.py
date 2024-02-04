import jax.numpy as jnp

class TriangleMesh():
    def __init__(self, node, cell):
        self.node = node
        self.cell = cell

    def number_of_cells(self):
        return len(self.cell)

    def number_of_nodes(self):
        return len(self.node)
