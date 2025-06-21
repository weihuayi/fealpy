import numpy as np

def gmsh_to_fealpy(mol, cls, dim, NVC):
    # get mesh information
    node = mol.mesh.get_nodes()[1].reshape(-1, 3)[:, :dim]
    NN = node.shape[0]

    nid2tag = mol.mesh.get_nodes()[0]
    tag2nid = np.zeros(NN*2, dtype = np.int_)
    tag2nid[nid2tag] = np.arange(NN)

    cell = mol.mesh.get_elements(dim, -1)[2][0].reshape(-1, NVC)
    cell = tag2nid[cell]

    # Construct FEALPy mesh
    mesh = cls(node, cell)

    # Get node tag and dim
    dims = np.zeros(NN)
    tags = np.zeros(NN)
    for i in range(dim+1)[::-1]:
        dimtags = mol.getEntities(i)
        for dim, tag in dimtags:
            idx = mol.mesh.get_elements(dim, tag)[2]
            if(len(idx)>0):
                idx = tag2nid[idx[0]]
                tags[idx] = tag
                dims[idx] = i

    mesh.nodedata['tag'] = tags
    mesh.nodedata['dim'] = dims
    mesh.celldata['z'] = mesh.entity_barycenter('cell')[:, -1]
    return mesh


