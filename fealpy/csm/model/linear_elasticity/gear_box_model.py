from typing import Optional

from ....backend import bm
from ....typing import TensorLike
from ....decorator import cartesian

from fealpy.mesh import (
        TetrahedronMesh,
        InpFileParser,
        )

from fealpy.material import (
        LinearElasticMaterial,
        )

class GearBoxModel:

    def __init__(self, options):
        self.options = options
        self.parser = InpFileParser()
        self.parser.parse(options['mesh_file'])
        self.material = self.parser.to_material(LinearElasticMaterial, 'gearbox')

    def __str__(self) -> str:
        """
        Return a human-readable summary of the model configuration.
        """
        mesh_file = self.options.get('mesh_file', '<unset>')
        return (
            f"GearBoxModel(\n"
            f"  mesh_file = '{mesh_file}',\n"
            f"  dimension = 3D,\n"
            f")"
        )

    def geo_dimension(self):
        return 3

    def init_mesh(self):
        """
        """
        return self.parser.to_mesh(TetrahedronMesh)
    
    