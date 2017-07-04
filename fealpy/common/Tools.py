
import numpy as np

def ranges(nv, start = 0):
    shifts = np.cumsum(nv)
    id_arr = np.ones(shifts[-1], dtype=np.int)
    id_arr[shifts[:-1]] = -np.asarray(nv[:-1])+1
    id_arr[0] = start 
    return id_arr.cumsum()
