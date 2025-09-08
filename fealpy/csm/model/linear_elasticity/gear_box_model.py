from typing import Optional

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from fealpy.material import (
        LinearElasticMaterial,
        )
from fealpy.mesh import (
        MeshData,
        TetrahedronMesh,
        InpFileParser,
        )

class GearBoxModel:
    """
    A model class representing a gearbox structure.

    This class loads mesh and material data from an ABAQUS `.inp` file using the
    `InpFileParser`, and provides access to geometry dimension and initialized mesh.

    Parameters:
        options (dict): A dictionary containing model configuration options.
            Required key:
                'mesh_file' (str): Path to the input `.inp` file describing the geometry and material.

    Attributes:
        options (dict): Configuration dictionary passed to the constructor.
        parser (InpFileParser): Internal parser used to read the `.inp` file and extract sections.
        material (LinearElasticMaterial): Material model parsed from the input file.

    Methods:
        __str__() -> str:
            Returns a human-readable summary of the model configuration.

        geo_dimension() -> int:
            Returns the spatial dimension of the model (always 3 for gearbox).

        init_mesh() -> TetrahedronMesh:
            Builds and returns a mesh object initialized from the `.inp` file.
    """

    def __init__(self, options):
        self.options = options
        self.parser = InpFileParser()
        self.parser.parse(options['mesh_file'])
        self.mesh = self.parser.to_mesh(TetrahedronMesh, MeshData)
        for name, data in self.mesh.data['materials'].items():
            self.material = LinearElasticMaterial(name, **data) 

    def __str__(self) -> str:
        """
        Return a human-readable summary of the model configuration.
        """
        mesh_file = self.options.get('mesh_file', '<unset>')
        return (
            f"\nGearBoxModel(\n"
            f"  mesh_file = '{mesh_file}',\n"
            f"  dimension = 3D,\n"
            f")"
        )

    def geo_dimension(self):
        return 3

    def init_mesh(self):
        """
        """
        return self.mesh 
    
    
