
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Dipole3d"]


class Dipole3d(CNodeType):
    r"""Create a mesh in a dipole-shaped 3D area.

    Inputs:
    cr (float): The radius of the cylinder.
    sr0 (float): The inner radius of the sphere.
    sr1 (float): The outer radius of the sphere.
    L (float): The length of the cylinder.
    G (float): The gap between the cylinder and the sphere.

    Outputs:
        mesh (MeshType): The mesh object created.
        ID1 (bool): The Robin boundary flag.
        dirichletBC (bool): The Dirichlet boundary ID.
    """
    TITLE: str = "偶极子天线网格"
    PATH: str = "网格.构造"
    INPUT_SLOTS = [
        PortConf("cr", DataType.FLOAT, 0, title="圆柱半径", default= 0.05),
        PortConf("sr0", DataType.FLOAT, 1, title="球形计算域内半径", default=1.9),
        PortConf("sr1", DataType.FLOAT, 1, title="球形计算域外半径", default=2.4),
        PortConf("L", DataType.FLOAT, 1, title="圆柱长度", default=2.01),
        PortConf("G", DataType.FLOAT, 1, title="缝隙长度", default=0.01)
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("ID1", DataType.BOOL, title="Robin边界"),
        PortConf("ID2", DataType.BOOL, title="Dirichlet边界")
    ]   
   
    @staticmethod
    def run(cr, sr0, sr1, L, G): 
        import gmsh
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import TetrahedronMesh

        # Initialize gmsh
        gmsh.initialize()

        # Create a new model
        model = gmsh.model

        # Set the model's dimension to 3D
        model.add("3D")

        # Construct model
        cyl0 = model.occ.addCylinder(0, 0, -L/2, 0, 0, L, cr)
        cyl1 = model.occ.addCylinder(0, 0, -G/2, 0, 0, G, cr)
        cyl  = model.occ.fragment([(3, cyl0)], [(3, cyl1)])[0]

        sphere0 = model.occ.addSphere(0, 0, 0, sr0)
        sphere1 = model.occ.addSphere(0, 0, 0, sr1)
        sphere = model.occ.fragment([(3, sphere1)], [(3, sphere0)])[0] 

        sphere = model.occ.cut(sphere, cyl)[0]

        box0 = model.occ.addBox(0, -3, -3,  6, 6, 6) 
        box1 = model.occ.addBox(0, -3, -3, -6, 6, 6) 
        box = model.occ.fragment([(3, box0)], [(3, box1)])[0]
        model.occ.intersect(box, sphere) 

        # Synchronize
        gmsh.model.occ.synchronize()

        # Construct size field
        Gamma0 = [2, 3, 15]                             
        Gamma1 = [10, 14, 19]                           
        Gamma2 = [7, 8, 9, 11, 12, 13, 17, 18, 20, 21]

        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "FacesList", Gamma1)
        model.mesh.field.setNumber(1, "Sampling", 100)

        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "InField", 1)
        model.mesh.field.setNumber(2, "DistMin", 0.05)
        model.mesh.field.setNumber(2, "DistMax", 0.2)
        model.mesh.field.setNumber(2, "SizeMin", 0.05)
        model.mesh.field.setNumber(2, "SizeMax", 0.18)

        model.mesh.field.add("Distance", 3)
        model.mesh.field.setNumbers(3, "FacesList", Gamma2)
        model.mesh.field.setNumber(3, "Sampling", 100)

        model.mesh.field.add("Threshold", 4)
        model.mesh.field.setNumber(4, "InField", 3)
        model.mesh.field.setNumber(4, "DistMin", 0.05)
        model.mesh.field.setNumber(4, "DistMax", 0.2)
        model.mesh.field.setNumber(4, "SizeMin", 0.05)
        model.mesh.field.setNumber(4, "SizeMax", 0.18)

        model.mesh.field.add("Constant", 5)
        model.mesh.field.setNumbers(5, "VolumesList", [1, 3])
        model.mesh.field.setNumber(5, "VIn", 0.03)

        model.mesh.field.add("Min",6)
        model.mesh.field.setNumbers(6, "FieldsList",[5, 2, 4]) 

        model.mesh.field.setAsBackgroundMesh(4)
        model.mesh.generate(3)

        node = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)
        NN = node.shape[0]

        nid2tag = gmsh.model.mesh.get_nodes()[0] 
        tagmax = int(bm.max(nid2tag))
        tag2nid = bm.zeros(tagmax+1, dtype = bm.int32)
        tag2nid[nid2tag] = bm.arange(NN)

        cell = gmsh.model.mesh.get_elements(3, -1)[2][0].reshape(-1, 4)
        cell = tag2nid[cell]

        # Construct FEALPy mesh
        mesh = TetrahedronMesh(node, cell)

        # Get node tag and dim
        dims = bm.zeros(NN)
        tags = bm.zeros(NN)
        for i in range(4)[::-1]:
            dimtags = model.getEntities(i)
            print(dimtags)
            for dim, tag in dimtags:
                idx = gmsh.model.mesh.get_elements(dim, tag)[2]
                if(len(idx)>0):
                    idx = tag2nid[idx[0]]
                    tags[idx] = tag
                    dims[idx] = i
                else:
                    print(dim, tag)

        mesh.nodedata['tag'] = tags
        mesh.nodedata['dim'] = dims
        mesh.celldata['z'] = mesh.entity_barycenter('cell')[:, -1]

        # Get face type
        NF = mesh.number_of_faces()
        faceint = bm.zeros(NF, dtype=bm.int32)
        face = mesh.entity('face')
        facetag = {'RobinFace'      : [8, 18, 7, 9, 17, 21, 12, 20, 11, 13],
                'DirichletFace' : [3, 2, 15, 14, 10, 19]}
        i = 0
        for key in facetag:
            lface = face.copy()
            for tag in facetag[key]:  
                idx = gmsh.model.mesh.get_elements(2, tag)[2]  
                idx = tag2nid[idx[0]].reshape(-1, 3) 
                # lface = bm.r_[lface, idx]  
                lface = bm.concatenate((lface, idx))  
            _, idx, jdx = bm.unique(bm.sort(lface, axis=1), return_index=True, 
                                    return_inverse=True, axis=0)  
            mesh.meshdata[key] = idx[jdx[NF:]]  
            faceint[idx[jdx[NF:]]] = i+2  
            i += 1


        mesh.facedata['attributes'] = faceint

        cellbar = mesh.entity_barycenter('cell')
        flag = bm.linalg.norm(cellbar, axis=-1) > sr0
        mesh.celldata['attributes'] = flag.astype(bm.int32)+1


        robinBC, dirichletBC = mesh.meshdata['RobinFace'], mesh.meshdata['DirichletFace'] 
        NF = mesh.number_of_faces()
        ID1 = bm.zeros(NF, dtype=bool)
        ID1[robinBC] = True

        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        gdof = NE * 2 + NF * 2
        ID2 = bm.zeros(gdof, dtype=bool)
        ID2[dirichletBC] = True

        return mesh, ID1, ID2

    