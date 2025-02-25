cdef extern from "cblas.h":
    void cblas_daxpy(int N, double alpha, double *X, int incX, double *Y, int incY)

def daxpy(int N, double alpha, double[:] X, int incX, double[:] Y, int incY):
     if N > X.shape[0] or N > Y.shape[0]:
        raise ValueError("向量长度 N 大于输入向量的实际长度")
    cblas_daxpy(N, alpha, &X[0], incX, &Y[0], incY)
