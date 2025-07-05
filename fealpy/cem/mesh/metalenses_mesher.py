from fealpy.backend import backend_manager as bm
from ...decorator import variantmethod


class MetalensesMesher:
    """
    The MetalensesMesher class is responsible for generating meshes for metalenses.
    It supports different mesh types, currently 'tet' for tetrahedral meshes, and   'pri' for prism meshes.

    Parameters:
        metalenses_params: dict
            A dictionary containing parameters for the metalenses,
            including base size, glass height, air layer height,
            bottom PML height, top PML height, and antenna sizes and heights.
        mesh_type: str
            The type of mesh to be generated. Default is 'tet' for tetrahedral meshes.
    """
    def __init__(self, metalenses_params, mesh_type='tet'):
        self.metalenses_params = metalenses_params
        self.generate.set(mesh_type)

    @variantmethod('tet')
    def generate(self, mesh_size=0.2):
        """
        Generate a tetrahedral mesh for one unit of metalenses.

        Parameters:
            mesh_size: float
                The size of the mesh elements. Default is 0.2.
        """
        from .metalenses_meshr_tet import MetalensesMesherTet

        mesher = MetalensesMesherTet(self.metalenses_params)
        return mesher.generate_mesh(mesh_size)

    @generate.register('pri')
    def generate(self, mesh_size=0.1):
        raise NotImplementedError("The 'pri' mesh type is not implemented yet.")

    def assemble_total_mesh(self, unit_mesh: None, mesh_type='tet'):
        """
        Assemble the total mesh from the individual components.
        maybe this method can be a static method,
        the only difference is that the number of vertices of one cell.

        Parameters:
            unit_mesh: Mesh
                The mesh of a single component, which can be a tetrahedral or prism mesh.
            mesh_type: str
                The type of mesh to be assembled. Default is 'tet' for tetrahedral meshes.
        """
        if unit_mesh is None:
            unit_mesh = self.generate['mesh_type']()

        if mesh_type == 'tet':
            pass
        if mesh_type == 'pri':
            pass

        raise NotImplementedError("This method should be implemented in subclasses.")






if __name__ == "__main__":
    from fealpy.mesh import TetrahedronMesh
    import json


    # # json 参数读取
    # with open("../data/parameters.json", "r") as f:
    #     metalenses_params = json.load(f)
    #
    # base_size = metalenses_params["base_size"]
    # glass_height = metalenses_params["glass_height"]
    # air_layer_height = metalenses_params["air_layer_height"]
    # bottom_pml_height = metalenses_params["bottom_pml_height"]
    # top_pml_height = metalenses_params["top_pml_height"]
    # antenna1_size = metalenses_params["antenna1_size"]
    # antenna1_height = metalenses_params["antenna1_height"]
    # antenna2_size = metalenses_params["antenna2_size"]
    # antenna2_height = metalenses_params["antenna2_height"]
    # antenna3_size = metalenses_params["antenna3_size"]
    # antenna3_height = metalenses_params["antenna3_height"]
    # antenna4_size = metalenses_params["antenna4_size"]
    # antenna4_height = metalenses_params["antenna4_height"]

    # 手动设置参数
    metalenses_params = {
        "base_size": 800,
        "glass_height": 3000,
        "air_layer_height": 4800,
        "bottom_pml_height": 960,
        "top_pml_height": 960,
        "antenna1_size": 190,
        "antenna1_height": 600,
        "antenna2_size": 160,
        "antenna2_height": 600,
        "antenna3_size": 160,
        "antenna3_height": 600,
        "antenna4_size": 160,
        "antenna4_height": 600
    }
    mesh_size = 0.1

    # 创建 MetalensesMesher 实例
    mesher = MetalensesMesher(metalenses_params)
    # 生成单个组件网格
    unit_tet_mesh = mesher.generate(mesh_size)
    # unit_tet_mesh.to_vtk(fname="../data/metalenses_tet_mesh.vtu")
