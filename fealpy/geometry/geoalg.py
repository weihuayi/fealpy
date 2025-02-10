
from ..backend import backend_manager as bm

def msign(x, eps=1e-10):
    flag = bm.sign(x)
    flag[bm.abs(x) < eps] = 0
    return flag

def find_cut_point(phi, p0, p1):
    """ 
    @brief Find cutted point between edge `(p0, p1)` and the curve `phi`

    """
    cutPoint = (p0+p1)/2.0
    phi0 = phi(p0)
    phi1 = phi(p1)
    phic = phi(cutPoint)

    isLeft = bm.zeros(p0.shape[0], dtype=bm.bool)
    isRight = bm.zeros(p0.shape[0], dtype=bm.bool_)
    vec = p1 - p0
    h = bm.sqrt(bm.sum(vec**2, axis=1))

    eps = bm.finfo(p0.dtype).eps
    tol = bm.sqrt(eps)*h*h
    isNotOK = (h > tol) & (phic != 0)
    while bm.any(isNotOK):
        cutPoint[isNotOK, :] = (p0[isNotOK, :] + p1[isNotOK,:])/2
        phic[isNotOK] = phi(cutPoint[isNotOK, :])
        isLeft[isNotOK] = phi0[isNotOK] * phic[isNotOK] > 0
        isRight[isNotOK] = phi1[isNotOK] * phic[isNotOK] > 0
        p0[isLeft, :] = cutPoint[isLeft, :]
        p1[isRight, :] = cutPoint[isRight, :]

        phi0[isLeft] = phic[isLeft]
        phi1[isRight] = phic[isRight]
        h[isNotOK] /= 2
        isNotOK[isNotOK] = (h[isNotOK] > tol[isNotOK]) & (phic[isNotOK] != 0)
        isLeft[:] = False
        isRight[:] = False
    return cutPoint



def project(imfun, p0, maxit=200, tol=1e-13, returngrad=False, returnd=False):

    eps = bm.finfo(float).eps
    p = p0
    value = imfun(p)
    s = bm.sign(value)
    grad = imfun.gradient(p)
    lg = bm.sum(grad**2, axis=-1, keepdims=True)
    grad /= lg
    grad *= value[..., None]
    pp = p - grad
    v = s[..., bm]*(pp - p0)
    d = bm.sqrt(bm.sum(v**2, axis=-1, keepdims=True))
    d *= s[..., None]

    g = imfun.gradient(pp)
    g /= bm.sqrt(bm.sum(g**2, axis=-1, keepdims=True))
    g *= d
    p = p0 - g

    k = 0
    while True:
        value = imfun(p)
        grad = imfun.gradient(p)
        lg = bm.sqrt(bm.sum(grad**2, axis=-1, keepdims=True))
        grad /= lg

        v = s[..., bm.newaxis]*(p0 - p)
        d = bm.sqrt(bm.sum(v**2, axis=-1))
        isOK = d < eps
        d[isOK] = 0
        v[isOK] = grad[isOK]
        v[~isOK] /= d[~isOK][..., None]
        d *= s

        ev = grad - v
        e = bm.max(bm.sqrt((value/lg.reshape(lg.shape[0:-1]))**2 + bm.sum(ev**2, axis=-1)))
        if e < tol:
            break
        else:
            k += 1
            if k > maxit:
                break
            grad /= lg
            grad *= value[..., None]
            pp = p - grad
            v = s[..., None]*(pp - p0)
            d = bm.sqrt(bm.sum(v**2, axis=-1, keepdims=True))
            d *= s[..., None]

            g = imfun.gradient(pp)
            g /= bm.sqrt(bm.sum(g**2, axis=-1, keepdims=True))
            g *= d
            p = p0 - g

    if (returnd is True) and (returngrad is True):
        return p, d, grad
    elif (returnd is False) and (returngrad is True):
        return p, grad
    elif (returnd is True) and (returngrad is False):
        return p, d
    else:
        return p
