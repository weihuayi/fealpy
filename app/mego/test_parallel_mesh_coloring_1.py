#!/usr/bin/env python3

import sys
import numpy as np
from meshpy.triangle import MeshInfo, build
from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.graph import metis
import matplotlib.pyplot as plt
from fealpy.mesh.simple_mesh_generator import squaremesh

from mpi4py import MPI


def get_ods(lmesh, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    NN = lmesh.number_of_nodes()

    edge = lmesh.entity('edge')
    parts = lmesh.nodedata['partition'] 
    gid = lmesh.nodedata['global_id']
    isLocalNode = (parts == rank)

    ods = {}
    if size > 1:
        # sent the local and global idx of the ghost nodes
        for prank in lmesh.neighbor:
            isGhostNode = (parts == prank)
            NG = isGhostNode.sum()
            sd = np.zeros((NG, 2), dtype=np.int)
            sd[:, 1], = np.nonzero(isGhostNode)
            sd[:, 0] = gid[sd[:, 1]]
            comm.Isend(sd, dest=prank, tag=rank)

        # recv the local and global idx of the local node as other process's ghost 
        for prank in lmesh.neighbor:
            isGhostNode = (parts == prank)
            isEdge = (np.sum(isGhostNode[edge], axis=1) == 1) & (np.sum(isLocalNode[edge], axis=1) == 1)
            isGNLNode = np.zeros(NN, dtype=np.bool)
            isGNLNode[edge[isEdge]] = True
            isGNLNode[isGhostNode] = False
            NR = np.sum(isGNLNode)
            ods[prank] = np.zeros((NR, 2), dtype=np.int)
            req = comm.Irecv(ods[prank], source=prank, tag=prank)
            req.Wait()
            ods[prank][:, 0] = lmesh.global2local[ods[prank][:, 0]]

    return ods

def par_regmesh(n):
    tmesh = squaremesh(0,1,0,1,3)

    # Partition the mesh cells into n parts 
    if n > 1:
        edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='node')
    else:
        NN = tmesh.number_of_nodes()
        parts = np.zeros(NN, dtype=np.int)
    tmesh.nodedata['partition'] = parts
    return tmesh


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

    # Generate the triangle mesh
    mesh = build(mesh_info, max_volume=(h)**2)

    node = np.array(mesh.points, dtype=np.float)
    cell = np.array(mesh.elements, dtype=np.int)

    tmesh = TriangleMesh(node, cell)

    # Partition the mesh cells into n parts 
    if n > 1:
        edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='node')
    else:
        NN = tmesh.number_of_nodes()
        parts = np.zeros(NN, dtype=np.int)
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

def set_ghost_random(r, isUnColor, isLocalNode, ods, lmesh, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    parts = lmesh.nodedata['partition'] 

    if size > 1:
        flag = isUnColor & isLocalNode
        for prank in lmesh.neighbor:
            rr = r[ods[prank][:, 0]]
            ff = flag[ods[prank][:, 0]]
            n = np.sum(ff)
            if n > 0:
                sd = np.zeros((n, 2), dtype=np.float) 
                sd[:, 0] = ods[prank][ff, 1]
                sd[:, 1] = rr[ff]
                comm.Isend(sd, dest=prank, tag=rank)

        for prank in lmesh.neighbor:
            isRecv = (parts == prank) & isUnColor
            n = np.sum(isRecv)
            if n > 0:
                rd = np.zeros((n, 2), dtype=np.float)
                req = comm.Irecv(rd, source=prank, tag=prank)
                req.Wait()
                r[rd[:, 0].astype(np.int)] = rd[:,  1]

def set_ghost_color(c, isLocalNode, ods, lmesh, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    parts = lmesh.nodedata['partition'] 
    isUnColor = (c == 0)


    if size > 1:
        for prank in lmesh.neighbor:
            cc = c[ods[prank][:, 0]]
            comm.Isend(c[ods[prank][:, 0]], dest=prank, tag=rank)

        for prank in lmesh.neighbor:
            isGhostNode = (parts == prank)
            isRecv = isGhostNode & isUnColor
            n = np.sum(isRecv)
            if n > 0:
                NG = np.sum(isGhostNode)
                rd = np.zeros(NG, dtype=np.int)
                req = comm.Irecv(rd, source=prank, tag=prank)
                req.Wait()
                c[isGhostNode] = rd


def coloring(lmesh, comm):
    rank = comm.Get_rank()
    parts = lmesh.nodedata['partition'] 
    NN = lmesh.number_of_nodes()

    isLocalNode = (parts == rank)
    c = np.zeros(NN, dtype=np.int)
    isUnColor = (c == 0) 

    color = 0
    ods = get_ods(lmesh, comm) 
    #np.random.seed(rank)

    r = np.zeros(NN, dtype=np.float)

    edge = lmesh.entity('edge')
    nc = np.zeros(1, dtype=np.int)
    totalnc = np.zeros(1, dtype=np.int) 
    while True:
        nc[0] = np.sum(isUnColor & isLocalNode)
        totalnc[0] = 0
        comm.Allreduce(nc, totalnc, op=MPI.SUM)

        if totalnc[0] == 0:
            break
        color += 1
        isRemainEdge = isUnColor[edge[:, 0]] & isUnColor[edge[:, 1]]
        edge0 = edge[isRemainEdge]

        r[:] = 0
        N = np.sum(isUnColor & isLocalNode)
        r[isUnColor & isLocalNode]  = np.random.rand(N)
        set_ghost_random(r, isUnColor, isLocalNode, ods, lmesh, comm) 

        isLess =  r[edge0[:, 0]] < r[edge0[:, 1]]
        flag = np.bincount(edge0[~isLess, 0], minlength=NN)
        flag += np.bincount(edge0[isLess, 1], minlength=NN)
        flag = (flag == 0) & isUnColor & isLocalNode
        c[flag] = color

        set_ghost_color(c, isLocalNode, ods, lmesh, comm) 
        isUnColor = (c == 0)


        isEdge0 = isUnColor[edge[:, 0]] & (c[edge[:, 1]] == color)
        isEdge1 = isUnColor[edge[:, 1]] & (c[edge[:, 0]] == color)
        flag = np.bincount(edge[isEdge0, 0], minlength=NN)
        flag += np.bincount(edge[isEdge1, 1], minlength=NN)
        isInteriorUnColor = (flag == 0)  & isUnColor 
        while True:
            nc[0] = np.sum(isInteriorUnColor & isLocalNode)
            totalnc[0] = 0
            comm.Allreduce(nc, totalnc, op=MPI.SUM)

            if totalnc[0] == 0:
                break

            isRemainEdge = isInteriorUnColor[edge[:,0]] & isInteriorUnColor[edge[:,1]]
            edge0 = edge[isRemainEdge]

            r[:] = 0
            N = np.sum(isUnColor & isLocalNode)

            r[isUnColor & isLocalNode]  = np.random.rand(N)
            set_ghost_random(r, isUnColor, isLocalNode, ods, lmesh, comm) 

            isLess =  r[edge0[:, 0]] < r[edge0[:, 1]]
            flag = np.bincount(edge0[~isLess, 0], minlength=NN)
            flag += np.bincount(edge0[isLess, 1], minlength=NN)
            c[(flag == 0) & isInteriorUnColor & isLocalNode] = color
            set_ghost_color(c, isLocalNode, ods, lmesh, comm) 
            isUnColor = (c == 0)


            isEdge0 = isUnColor[edge[:, 0]] & (c[edge[:, 1]] == color)
            isEdge1 = isUnColor[edge[:, 1]] & (c[edge[:, 0]] == color)
            flag = np.bincount(edge[isEdge0, 0], minlength=NN)
            flag += np.bincount(edge[isEdge1, 1], minlength=NN)
            isInteriorUnColor = (flag==0) & isUnColor 

    for i in range(1, color+1):
        print('There are {} points with color {} in rank {}.'.format((c==i).sum(), i, rank))

    return c

def check_color(lmesh, c):
    edge = lmesh.entity('edge')
    flag = (c[edge[:, 0]] == c[edge[:, 1]])
    return flag 

def check_parallel_data(lmesh, ods, comm): 
    rank = comm.Get_rank()
    fig = plt.figure()
    axes = fig.gca()
    lmesh.add_plot(axes, cellcolor='w')
    lmesh.find_node(axes, color=lmesh.nodedata['partition'], markersize=20,
            showindex=True)
    for prank in lmesh.neighbor:
        print("Process ", rank, " and  process ", prank, " with data \n", ods[prank]) 

    axes.set_title("rank {}".format(rank))

def show_mesh(lmesh, data):
    fig = plt.figure()
    axes = fig.gca()
    lmesh.add_plot(axes, cellcolor='w')
    lmesh.find_node(axes, color=lmesh.nodedata['partition'], markersize=20)
    NN = lmesh.number_of_nodes()
    node = lmesh.entity('node')
    for i in range(NN):
        axes.text(node[i, 0], node[i, 1], str(data[i])[0:3],
                multialignment='center', fontsize=10) 
    axes.set_title("rank {}".format(rank), fontsize=20)
    return axes

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    print('Size:', size)
    print('Rank:', rank)
    print('Name:', name)

    tmesh = par_regmesh(size)
    #tmesh = par_mesh(0.05, size) 
    lmesh = local_mesh(tmesh, rank)

    t0 = MPI.Wtime()
    c = coloring(lmesh, comm)
    t1 = MPI.Wtime()
    print('color time', t1-t0)

    flag = check_color(lmesh, c)
    print('Process ', rank, " with same coloring ",  np.sum(flag))

    axes = show_mesh(lmesh, c)
    #lmesh.find_edge(axes, index=flag) 
    ##show_mesh(lmesh, r1)
    plt.show()
