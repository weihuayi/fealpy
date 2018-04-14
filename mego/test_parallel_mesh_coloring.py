#!/usr/bin/env python3

import sys
import numpy as np
from meshpy.triangle import MeshInfo, build
from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.graph import metis
import matplotlib.pyplot as plt

from mpi4py import MPI


class MeshOverlapDataStructure():
    def __init__(self, lmesh, comm):
        self.send = {}
        self.recv = {}

        rank = comm.Get_rank()
        NN = lmesh.number_of_nodes()

        edge = lmesh.entity('edge')
        parts = lmesh.nodedata['partition'] 
        gid = lmesh.nodedata['global_id']
        isLocalNode = (parts == rank)
        for prank in lmesh.neighbor:
            isRecvNode = (parts == prank)
            isEdge = (np.sum(isRecvNode[edge], axis=1) == 1) & (np.sum(isLocalNode[edge], axis=1) == 1)
            isSentNode = np.zeros(NN, dtype=np.bool)
            isSentNode[edge[isEdge]] = True
            isSentNode[isRecvNode] = False
            NS = np.sum(isSentNode)
            self.send[prank], = np.nonzero(isSentNode) 
            comm.Isend(gid[isSentNode], dest=prank, tag=rank)

        for prank in lmesh.neighbor:
            isRecvNode = (parts == prank)
            NR = np.sum(isRecvNode)
            rd = np.zeros(NR, dtype=np.int)
            req = comm.Irecv(rd, source=prank, tag=prank)
            req.Wait()
            self.recv[prank] = lmesh.global2local[rd]

def set_gost_node(data, pds, lmesh, comm):
    rank = comm.Get_rank()
    for prank in lmesh.neighbor:
        sd = data[pds.send[prank]]
        comm.Isend(sd, dest=prank, tag=rank)

    for prank in lmesh.neighbor:
        NR = len(pds.recv[prank])
        rd = np.zeros(NR, dtype=np.int)
        req = comm.Irecv(rd, source=prank, tag=prank)
        req.Wait()
        data[pds.recv[prank]] = rd



def par_mesh(h, n):
    mesh_info = MeshInfo()

    # Set the vertices of the domain [0, 1]^2
    mesh_info.set_points([
        (0,0), (1,0), (1,1), (0,1)])

    # Set the facets of the domain [0, 1]^2
    mesh_info.set_facets([
        [0,1],
        [1,2],
        [2,3],
        [3,0]
        ])

    # Generate the tet mesh
    mesh = build(mesh_info, max_volume=(h)**2)

    node = np.array(mesh.points, dtype=np.float)
    cell = np.array(mesh.elements, dtype=np.int)

    tmesh = TriangleMesh(node, cell)

    # Partition the mesh cells into n parts 
    edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='node')
    tmesh.nodedata['partition'] = parts
    return tmesh


def local_mesh(tmesh, rank):
    NN = tmesh.number_of_nodes()

    parts = tmesh.nodedata['partition'] 

    cell = tmesh.entity('cell')
    node = tmesh.entity('node')

    flag = np.sum(parts[cell] == rank, axis=1) > 0

    cell0 = cell[flag]
    isLocalNode = np.zeros(NN, dtype=np.bool) 
    isLocalNode[cell0] = True
    node0 = node[isLocalNode]
    gid, = np.nonzero(isLocalNode)


    # build the idx map from global to local 
    NN0 = np.sum(isLocalNode)
    idxMap = np.zeros(NN, dtype=np.int)
    idxMap[isLocalNode] = range(NN0)

    cell0 = idxMap[cell0] 
    lmesh = TriangleMesh(node0, cell0)
    lmesh.nodedata['partition'] = parts[isLocalNode]
    lmesh.nodedata['global_id'] = gid 
    lmesh.global2local = idxMap

    lmesh.neighbor = set(parts[isLocalNode]) - {rank}
    return lmesh


def coloring(lmesh, comm):
    NN = lmesh.number_of_nodes()
    c = np.zeros(NN, dtype=np.int)

    parts = lmesh.nodedata['partition'] 
    isUnColor = (c == 0) 
    color = 0

    pds = MeshOverlapDataStructure(lmesh, comm)

    r = np.random.randint(0, 100, NN)

    set_gost_node(r, pds, lmesh, comm) 

#    edge = lmesh.entity('edge')
#    parts = lmesh.nodedata['partition'] 
#    gid = lmesh.nodedata['global_id']
#    isLocalNode = (parts == rank)
#    for prank in lmesh.neighbor:
#        isRecvNode = (parts == prank)
#        isEdge = (np.sum(isRecvNode[edge], axis=1) == 1) & (np.sum(isLocalNode[edge], axis=1) == 1)
#        isSentNode = np.zeros(NN, dtype=np.bool)
#        isSentNode[edge[isEdge]] = True
#        isSentNode[isRecvNode] = False
#        NS = np.sum(isSentNode)
#        sd = np.zeros((NS, 2), dtype=np.int)  
#        sd[:, 0] = gid[isSentNode]
#        sd[:, 1] = r[isSentNode]
#        comm.Isend(sd, dest=prank, tag=rank)
#        print("Process ", rank, " sent data ", sd.shape, " to process ", prank)
#
#    for prank in lmesh.neighbor:
#        isRecvNode = (parts == prank)
#        NR = np.sum(isRecvNode)
#        rd = np.zeros((NR, 2), dtype=np.int)
#        req = comm.Irecv(rd, source=prank, tag=prank)
#        print("Process ", rank, " receive data ", rd.shape, " from process ", prank)
#        req.Wait()
#        idx = lmesh.global2local[rd[:, 0]]
#        r[idx] = rd[:, 1]

    return r
#    edge = mesh.ds.edge
#    isRemainEdge = isUnColor[edge[:, 0]] & isUnColor[edge[:, 1]] 
#    while np.any(isRemainEdge):
#        color += 1
#        edge0 = edge[isRemainEdge]
#        isLess =  r[edge0[:, 0]] < r[edge0[:, 1]]
#        flag = np.bincount(edge0[~isLess, 0], minlength=NN)
#        flag += np.bincount(edge0[isLess, 1], minlength=NN)
#        c[(flag == 0) & isUnColor] = color
#        isUnColor = (c == 0) 



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    print('Size:', size)
    print('Rank:', rank)
    print('Name:', name)

    tmesh = par_mesh(0.05, size) 
    lmesh = local_mesh(tmesh, rank)
    r = coloring(lmesh, comm)

    fig = plt.figure()
    axes = fig.gca()
    lmesh.add_plot(axes, cellcolor='w')
    lmesh.find_node(axes, color=lmesh.nodedata['partition'], markersize=20)
    NN = lmesh.number_of_nodes()
    node = lmesh.entity('node')
    for i in range(NN):
        axes.text(node[i, 0], node[i, 1], str(r[i]), multialignment='center') 

    axes.set_title("rank {}".format(rank))
    plt.show()
