
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

        self.coupling = self.parser.to_coupling()
        self.material = self.parser.to_material()
        self.boundary = self.parser.to_boundary()

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
    
    def elsets(self):
        """"""
        pass

    def nsets(self):
        """
        """
        pass

    def solid(self):
        """
        """
        pass

    def surface(self):
        """
        """
        pass

    def coupling(self):
        """
        """
        pass

    def density(self):
        """
        """
        den = self.material[0]['density']
        return den
    
    def young(self):
        """
        Young's Modulus
        """
        E = self.material[0]['elastic'][0]
        return E

    def poissonratio(self):
        '''
        '''
        v = self.material[0]['elastic'][1]
        return v
    
    def boundary(self):
        """
        """
        return self.parser.to_boundary()