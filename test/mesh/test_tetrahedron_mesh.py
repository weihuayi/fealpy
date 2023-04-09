import numpy as np
import pytest
from fealpy.mesh import TetrahedronMesh 

def test_uniform_refine():
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')

    vol = mesh.entity_measure('cell')

    mesh.uniform_refine()

    vol = mesh.entity_measure('cell')

    mesh.uniform_refine()

    vol = mesh.entity_measure('cell')

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    print(cell[0, :])
    print(node[cell[0]])



    print(vol)


if __name__ == "__main__":
    test_uniform_refine()
