import numpy as np
def writefile(A, b, n):
    out1 = f'./Amesh{n}.txt'
    out2 = f'./bmesh{n}.txt'
    n = np.array([A.shape[0]])
    indptr = A.indptr
    indices = A.indices
    data = A.data
    with open(out1, 'w') as f:
        np.savetxt(f, n, fmt='%d')
    with open(out1, 'a') as f:
        np.savetxt(f, indptr, fmt='%d')
        np.savetxt(f, indices, fmt='%d')
        np.savetxt(f, data, fmt='%0.17f')
    with open(out2, 'w') as f:
        np.savetxt(f, b, fmt='%0.17f')
    return
