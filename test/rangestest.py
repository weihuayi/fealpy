import numpy as np 

def ranges(nv, start = 0):
    shifts = np.cumsum(nv)
    id_arr = np.ones(shifts[-1], dtype=np.int)
    id_arr[shifts[:-1]] = -np.asarray(nv[:-1])+1
    id_arr[0] = start 
    return id_arr.cumsum()

def cell_idx_1(p):
    ldof = int((p+1)*(p+2)/2)
    idx = np.arange(0, ldof)
    idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
    idx1 = idx0*(idx0+1)/2
    cellIdx = np.zeros((ldof, 3), dtype=np.int)
    cellIdx[:,2] = idx - idx1 
    cellIdx[:,1] = idx0 - cellIdx[:,2]
    cellIdx[:,0] = p - cellIdx[:, 1] - cellIdx[:, 2] 

def cell_idx_2(p):
    ldof = int((p+1)*(p+2)/2)
    cellIdx = np.zeros((ldof, 3), dtype=np.int)
    cellIdx[:,0] = np.repeat(range(p, -1, -1), range(1, p+2))
    cellIdx[:,2] = ranges(range(1,p+2)) 
    cellIdx[:,1] = p - cellIdx[:, 0] - cellIdx[:,2] 



