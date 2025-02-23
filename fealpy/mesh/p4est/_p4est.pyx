
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from libc.stdlib cimport free
from mpi4py import MPI


cdef extern from "mpi.h":
    ctypedef struct ompi_communicator_t:
        pass
    ctypedef ompi_communicator_t *MPI_Comm
    ctypedef MPI_Comm    sc_MPI_Comm

    int MPI_Init(int *argc, char ***argv)
    int MPI_Initialized(int *flag)


##################################################
### Connectivity
##################################################

cdef extern from "p4est_connectivity.h":
    ctypedef int p4est_topidx_t
    ctypedef char int8_t

    ctypedef struct p4est_connectivity_t:
        p4est_topidx_t num_vertices
        p4est_topidx_t num_trees
        p4est_topidx_t num_corners
        double *vertices
        p4est_topidx_t *tree_to_vertex
        size_t tree_attr_bytes
        char *tree_to_attr
        p4est_topidx_t *tree_to_tree
        int8_t *tree_to_face
        p4est_topidx_t *tree_to_corner
        p4est_topidx_t *ctt_offset
        p4est_topidx_t *corner_to_tree
        int8_t *corner_to_corner

    p4est_connectivity_t *p4est_connectivity_new_unitsquare()
    void p4est_connectivity_destroy(p4est_connectivity_t *conn)


cdef void del_connectivity(object capsule) noexcept:
    cdef p4est_connectivity_t* conn
    conn = <p4est_connectivity_t*>PyCapsule_GetPointer(capsule, "conn_ptr")
    if conn != NULL:
        p4est_connectivity_destroy(conn)


### Properties

def get_num_vertices_py(object capsule):
    cdef p4est_connectivity_t* conn = get_conn_ptr(capsule)
    return conn.num_vertices if conn != NULL else 0

### Pointer and Shape of array

def get_vertices_ptr_py(object capsule):
    cdef p4est_connectivity_t* conn = get_conn_ptr(capsule)
    return <size_t>conn.vertices if conn != NULL else 0

def get_vertices_shape_py(object capsule):
    cdef p4est_connectivity_t* conn = get_conn_ptr(capsule)
    return (conn.num_vertices, 3) if conn != NULL else (<p4est_topidx_t>0, 3)

# Utility: Get pointer from capsule

cdef p4est_connectivity_t* get_conn_ptr(object capsule) except NULL:
    return <p4est_connectivity_t*>PyCapsule_GetPointer(capsule, "conn_ptr")


### Constructors

def p4est_conn_new_unitsquare_py():
    cdef p4est_connectivity_t* conn = p4est_connectivity_new_unitsquare()
    return PyCapsule_New(conn, "conn_ptr", del_connectivity)


##################################################
### Geometry
##################################################

cdef extern from "p4est_geometry.h":
    ctypedef struct p4est_geometry:
        pass
    ctypedef p4est_geometry p4est_geometry_t


##################################################
### P4est
##################################################

cdef extern from "p4est.h":
    ctypedef int            p4est_qcoord_t
    ctypedef int            p4est_topidx_t
    ctypedef int            p4est_locidx_t
    ctypedef long long      p4est_gloidx_t
    ctypedef p4est_connect_type_t

    ctypedef struct sc_array:
        pass
    ctypedef sc_array sc_array_t
    ctypedef struct sc_mempool:
        pass
    ctypedef sc_mempool sc_mempool_t

    ctypedef struct p4est_quadrant_t:
        pass
    ctypedef struct p4est_connectivity_t:
        pass
    ctypedef struct p4est_inspect_t:
        pass

    ctypedef struct p4est_t:
        sc_MPI_Comm         mpicomm          # MPI communicator
        int                 mpisize          # number of MPI processes
        int                 mpirank          # this process's MPI rank
        int                 mpicomm_owned    # flag if communicator is owned
        size_t              data_size        # size of per-quadrant p.user_data
                            # (see p4est_quadrant_t::p4est_quadrant_data::user_data)
        void               *user_pointer      # convenience pointer for users,
                                              # never touched by p4est

        long                revision         # Gets bumped on mesh change
        p4est_topidx_t      first_local_tree # 0-based index of first local
                                              # tree, must be -1 for an empty
                                              # processor
        p4est_topidx_t      last_local_tree   # 0-based index of last local
                                              # tree, must be -2 for an empty
                                              # processor
        p4est_locidx_t      local_num_quadrants    # number of quadrants on all
                                                   # trees on this processor
        p4est_gloidx_t      global_num_quadrants   # number of quadrants on all
                                                   # trees on all processors
        p4est_gloidx_t     *global_first_quadrant  # first global quadrant index
                                                   # for each process and 1
                                                   # beyond
        p4est_quadrant_t   *global_first_position  # first smallest possible quad
                                                   # for each process and 1
                                                   # beyond
        p4est_connectivity_t *connectivity # connectivity structure, not owned
        sc_array_t         *trees          # array of all trees

        sc_mempool_t       *user_data_pool # memory allocator for user data
            # WARNING: This is NULL if data size
            # equals zero.
        sc_mempool_t       *quadrant_pool   # memory allocator for temporary
                                            # quadrants
        p4est_inspect_t    *inspect         # algorithmic switches


    ctypedef void (*p4est_init_t) (p4est_t * p4est,
                                   p4est_topidx_t which_tree,
                                   p4est_quadrant_t * quadrant)

    ctypedef void (*p4est_refine_t) (p4est_t * p4est,
                                     p4est_topidx_t which_tree,
                                     p4est_quadrant_t * quadrant)

    ctypedef void (*p4est_coarsen_t) (p4est_t * p4est,
                                      p4est_topidx_t which_tree,
                                      p4est_quadrant_t * quadrants[])

    p4est_t *p4est_new (sc_MPI_Comm mpicomm,
                        p4est_connectivity_t * connectivity,
                        size_t data_size,
                        p4est_init_t init_fn, void *user_pointer)

    void p4est_destroy (p4est_t * p4est)

    p4est_t *p4est_copy (p4est_t * input, int copy_data)

    void p4est_refine (p4est_t * p4est,
                       int refine_recursive,
                       p4est_refine_t refine_fn, p4est_init_t init_fn)

    void p4est_coarsen (p4est_t * p4est,
                        int coarsen_recursive,
                        p4est_coarsen_t coarsen_fn, p4est_init_t init_fn)

    void p4est_balance (p4est_t * p4est,
                        p4est_connect_type_t btype,
                        p4est_init_t init_fn)


    # VTK
    void p4est_vtk_write_file (p4est_t * p4est,
                               p4est_geometry_t * geom,
                               const char *filename)


cdef void del_p4est(object capsule) noexcept:
    cdef p4est_t* p4est
    p4est = <p4est_t*>PyCapsule_GetPointer(capsule, "p4est_ptr")
    if p4est != NULL:
        p4est_destroy(p4est)


# Utility: Get pointer from capsule

cdef p4est_t * get_p4est_ptr(object p4est_cap) except NULL:
    return <p4est_t*>PyCapsule_GetPointer(p4est_cap, "p4est_ptr")

cdef void init_fn (p4est_t *p4est, p4est_topidx_t which_tree,
                   p4est_quadrant_t *quadrant) noexcept:
    pass


### Constructors

def p4est_new_py(py_comm_addr, conn_cap):
    cdef unsigned long comm_hdl = py_comm_addr
    cdef MPI_Comm* comm_ptr = <MPI_Comm*>comm_hdl
    cdef MPI_Comm comm = comm_ptr[0]
    cdef p4est_connectivity_t* conn_ptr = get_conn_ptr(conn_cap)
    cdef p4est_t* p4est_ptr = p4est_new(comm, conn_ptr, 0, init_fn, NULL)
    return PyCapsule_New(p4est_ptr, "p4est_ptr", del_p4est)


### Operations


### VTK
#def p4est_vtk_write_file_py(p4est_cap, geom_cap, vtk_file_name):
#    p4est_t* p4est_ptr = get_p4est_ptr(p4est_cap)
