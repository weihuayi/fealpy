import numpy as np

def project(imfun, p0, maxit=200, tol=1e-13, returngrad=False, returnd=False):

    eps = np.finfo(float).eps
    p = p0
    value = imfun(p)
    s = np.sign(value)
    grad = imfun.gradient(p)
    lg = np.sum(grad**2, axis=-1, keepdims=True)
    grad /= lg
    grad *= value[..., np.newaxis]
    pp = p - grad
    v = s[..., np.newaxis]*(pp - p0)
    d = np.sqrt(np.sum(v**2, axis=-1, keepdims=True))
    d *= s[..., np.newaxis]

    g = imfun.gradient(pp)
    g /= np.sqrt(np.sum(g**2, axis=-1, keepdims=True))
    g *= d
    p = p0 - g

    k = 0
    while True:
        value = imfun(p)
        grad = imfun.gradient(p)
        lg = np.sqrt(np.sum(grad**2, axis=-1, keepdims=True))
        grad /= lg

        v = s[..., np.newaxis]*(p0 - p)
        d = np.sqrt(np.sum(v**2, axis=-1))
        isOK = d < eps
        d[isOK] = 0
        v[isOK] = grad[isOK]
        v[~isOK] /= d[~isOK][..., np.newaxis]
        d *= s

        ev = grad - v
        e = np.max(np.sqrt((value/lg.reshape(lg.shape[0:-1]))**2 + np.sum(ev**2, axis=-1)))
        if e < tol:
            break
        else:
            k += 1
            if k > maxit:
                break
            grad /= lg
            grad *= value[..., np.newaxis]
            pp = p - grad
            v = s[..., np.newaxis]*(pp - p0)
            d = np.sqrt(np.sum(v**2, axis=-1, keepdims=True))
            d *= s[..., np.newaxis]

            g = imfun.gradient(pp)
            g /= np.sqrt(np.sum(g**2, axis=-1, keepdims=True))
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
