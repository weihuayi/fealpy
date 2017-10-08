import numpy as np

def mark(eta, theta, method='L2'):
    isMarked = np.zeros(len(eta), dtype=np.bool)
    if method is 'MAX':
        isMarked[eta > theta*np.max(eta)] = True
    elif method is 'L2':
        eta = eta**2
        idx = np.argsort(eta)[-1::-1]
        x = np.cumsum(eta[idx])
        isMarked[idx[x < theta*x[-1]]] = True
        isMarked[idx[0]] = True
    else:
        raise ValueError("I have not code the method")
    markedCell, = np.nonzero(isMarked)
    return markedCell
