from libc.stdlib cimport malloc, free

cdef extern from "pangulu_r64.h":

    ctypedef unsigned long long int sparse_pointer_t;
    ctypedef unsigned int sparse_index_t;
    ctypedef float sparse_value_t;

    ctypedef struct pangulu_init_options:
        int nthread
        int nb

    ctypedef struct pangulu_gstrf_options:
        pass

    ctypedef struct pangulu_gstrs_options:
        pass

    void pangulu_init(
            sparse_index_t n, 
            sparse_pointer_t nnz, 
            sparse_pointer_t *csr_rowptr, 
            sparse_index_t *csr_colidx, 
            sparse_value_t *csr_value, 
            pangulu_init_options *init_options, 
            void **handle);
    void pangulu_gstrf(
            pangulu_gstrf_options *gstrf_options, 
            void **handle);
    void pangulu_gstrs(
            sparse_value_t *rhs, 
            pangulu_gstrs_options *gstrs_options, 
            void **pangulu_handle);
    void pangulu_gssv(
            sparse_value_t *rhs, 
            pangulu_gstrf_options *gstrf_options, 
            pangulu_gstrs_options *gstrs_options, 
            void **pangulu_handle);
    void pangulu_finalize(void **pangulu_handle);


cdef class InitOptions:
    cdef pangulu_init_options _opts
    def __init__(self, int nthread=0, int nb=0):
        self._opts.nthread = nthread
        self._opts.nb = nb
    @property
    def nthread(self):
        return self._opts.nthread
    @nthread.setter
    def nthread(self, value):
        self._opts.nthread = value
    @property
    def nb(self):
        return self._opts.nb
    @nb.setter
    def nb(self, value):
        self._opts.nb = value

cdef class GstrfOptions:
    cdef pangulu_gstrf_options _opts  # 假设后续可能需要参数

cdef class GstrsOptions:
    cdef pangulu_gstrs_options _opts

cdef class Handle:
    cdef void* _handle
    def __dealloc__(self):
        if self._handle != NULL:
            pangulu_finalize(&self._handle)

def init(unsigned int n, 
         unsigned long long int nnz, 
         unsigned long long int[::1] csr_rowptr, 
         unsigned int[::1] csr_colidx, 
         float[::1] csr_value, 
         InitOptions opts not None):
    cdef Handle handle_obj = Handle()
    pangulu_init(n, nnz, &csr_rowptr[0], &csr_colidx[0], &csr_value[0], 
                &opts._opts, &handle_obj._handle)
    return handle_obj

def gstrf(GstrfOptions opts, Handle handle not None):
    """
    performs distribute sparse LU factorisation. Note that you should
    call pangulu_init() before calling pangulu_gstrf() to create a handle of PanguLU.
    """
    pangulu_gstrf(&opts._opts if opts is not None else NULL, &handle._handle)

def gstrs(float[::1] rhs, GstrsOptions opts, Handle handle not None):
    """
    solves linear equation with factorised L and U, and right-hand
    side vector b. Note that you should call pangulu_gstrf() before calling
    pangulu_gstrs() to ensure that L and U are available.
    """
    pangulu_gstrs(&rhs[0], &opts._opts if opts is not None else NULL, 
                 &handle._handle)

def gssv(float[::1] rhs, GstrfOptions gstrf_opts, GstrsOptions gstrs_opts, 
         Handle handle not None):
    """
    pangulu_gssv() solves the linear equation with A and right-hand size b. This
    function is equivalent to calling pangulu_gstrs() after pangulu_gstrf().
    """
    pangulu_gssv(&rhs[0], 
                &gstrf_opts._opts if gstrf_opts is not None else NULL,
                &gstrs_opts._opts if gstrs_opts is not None else NULL,
                &handle._handle)
