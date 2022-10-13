
import numpy as np

def ranges(nv, start = 0):
    shifts = np.cumsum(nv)
    id_arr = np.ones(shifts[-1], dtype=np.int)
    id_arr[shifts[:-1]] = -np.asarray(nv[:-1])+1
    id_arr[0] = start
    return id_arr.cumsum()

def hash2map(dec, ha):
    n = ha.shape[1]
    b = np.floor(dec.reshape(-1, 1)/2**np.arange(n))%2
    m = np.zeros_like(b)
    v = np.zeros_like(dec)
    idx, jdx = np.nonzero(b)
    for i in range(len(dec)):
        if dec[i]:
            flag = (idx==i)
            pos, _ = np.nonzero(ha[:, jdx[flag]])
            mdx = np.argmin(np.sum(np.abs(ha[pos] - b[i]), axis=-1))
            m[i] = ha[pos[mdx]]
            v[i] = pos[mdx]+1

    return m, v

def angle(v0, v1):
    a = v0/np.linalg.norm(v0, axis=-1)[:, None]
    b = v1/np.linalg.norm(v1, axis=-1)[:, None]
    cos = np.sum(a*b, axis=-1)
    if a.shape[-1]==2:
        sin = np.cross(a, b)
    elif a.shape[-1]==3:
        sin = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.arctan2(sin, cos)

