from ..backend import backend_manager as bm
from ..backend import TensorLike


def ranges(nv: TensorLike, start = 0):
    shifts = bm.cumsum(nv, axis=0)
    id_arr = bm.ones(shifts[-1], dtype=bm.int32)
    id_arr = bm.set_at(id_arr, (shifts[:-1]), -bm.asarray(nv[:-1])+1)
    id_arr = bm.set_at(id_arr, (0), start)
    return bm.cumsum(id_arr, axis=0)


def hash2map(dec: TensorLike, ha: TensorLike):
    n = ha.shape[1]
    b = bm.floor(dec.reshape(-1, 1)/2**bm.arange(n))%2
    m = bm.zeros_like(b)
    v = bm.zeros_like(dec)
    idx, jdx = bm.nonzero(b)
    for i in range(len(dec)):
        if dec[i]:
            flag = (idx==i)
            pos, _ = bm.nonzero(ha[:, jdx[flag]])
            mdx = bm.argmin(bm.sum(bm.abs(ha[pos] - b[i]), axis=-1))
            m = bm.set_at(m, (i), ha[pos[mdx]])
            v = bm.set_at(v, (i), pos[mdx]+1)

    return m, v


def angle(v0: TensorLike, v1: TensorLike):
    a = v0/bm.sqrt(bm.sum(v0**2, axis=-1, keepdims=True))
    b = v1/bm.sqrt(bm.sum(v1**2, axis=-1, keepdims=True))
    cos = bm.sum(a*b, axis=-1)
    if a.shape[-1]==2:
        sin = bm.cross(a, b, axis=-1)
    elif a.shape[-1]==3:
        c = bm.cross(a, b, axis=-1)
        sin = bm.sqrt(bm.sum(c**2, axis=-1))
    return bm.arctan2(sin, cos)
