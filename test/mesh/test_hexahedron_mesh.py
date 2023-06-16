import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh.hexahedron_mesh import HexahedronMesh



def test_hexahedrom_mesh_measure():
    mesh = HexahedronMesh.from_one_hexahedron()

    vol = mesh.entity_measure('cell')
    print(vol)

    area = mesh.entity_measure('face')
    print(area)



if __name__ == "__main__":
    test_hexahedrom_mesh_measure()
