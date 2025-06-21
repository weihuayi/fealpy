# Original code from pymumps (BSD 3-Clause License)
# Copyright (c) 2013, Bradley Froehle <brad.froehle@gmail.com>
# See: https://github.com/pymumps/pymumps

import warnings
import numpy as np
from . import _dmumps

__all__ = [
    'DMumpsContext',
    ]

########################################################################
# Classes
########################################################################

# The main class which is shared between the various datatype variants.
class _MumpsBaseContext(object):
    """MUMPS Context

    This context acts as a thin wrapper around MUMPS_STRUC_C
    which is accessible in the `id` attribute.

    Usage
    -----

    Basic usage generally involves setting up the context, adding
    the sparse matrix and right hand side in process 0, and using
    `run` to execute the various MUMPS phases.

        ctx = MumpsContext()
        if rank == 0:
            ctx.set_centralized_sparse(A)
            x = b.copy() # MUMPS modifies rhs in place, so make copy
            ctx.set_rhs(x)
        ctx.run(6) # Symbolic + Numeric + Solve
        ctx.destroy() # Free internal data structures

        assert abs(A.dot(x) - b).max() < 1e-10
    """

    def __init__(self, par=1, sym=0, comm=None):
        """Create a MUMPS solver context.

        Parameters
        ----------
        par : int
            1 if rank 0 participates
            0 if rank 0 does not participate
        sym : int
            0 if unsymmetric
        comm : MPI Communicator or None
            If None, use MPI_COMM_WORLD
        """
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

        self.id = self._MUMPS_STRUC_C()
        self.id.par = par
        self.id.sym = sym
        self.id.comm_fortran = comm.py2f()
        self.run(job = -1) # JOB_INIT
        self.myid = comm.rank
        self._refs = {} # References to matrices

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.destroy()

    def set_shape(self, n):
        """Set the matrix shape."""
        self.id.n = n

    def set_centralized_sparse(self, A):
        """Set assembled matrix on processor 0.

        Parameters
        ----------
        A : `scipy.sparse.coo_matrix`
            Sparse matrices of other formats will be converted to
            COOrdinate form.
        """
        if self.myid != 0:
            return

        A = A.tocoo()
        n = A.shape[0]
        assert A.shape == (n, n), "Expected a square matrix."
        self.set_shape(n)
        row = np.astype(np.from_dlpack(A.row+1), 'i')
        col = np.astype(np.from_dlpack(A.col+1), 'i')
        self.set_centralized_assembled(row, col, A.data)


    ####################################################################
    # Centralized (the rank 0 process supplies the entire matrix)
    ####################################################################

    def set_centralized_assembled(self, irn, jcn, a):
        """Set assembled matrix on processor 0.

        The row and column indices (irn & jcn) should be one based.
        """
        self.set_centralized_assembled_rows_cols(irn, jcn)
        self.set_centralized_assembled_values(a)

    def set_centralized_assembled_rows_cols(self, irn, jcn):
        """Set assembled matrix indices on processor 0.

        The row and column indices (irn & jcn) should be one based.
        """
        if self.myid != 0:
            return
        assert irn.size == jcn.size
        self._refs.update(irn=irn, jcn=jcn)
        self.id.nz = irn.size
        self.id.irn = self.cast_array(irn)
        self.id.jcn = self.cast_array(jcn)

    def set_centralized_assembled_values(self, a):
        """Set assembled matrix values on processor 0."""
        if self.myid != 0:
            return
        assert a.size == self.id.nz
        self._refs.update(a=a)
        self.id.a = self.cast_array(a)


    ####################################################################
    # Distributed (each process enters some portion of the matrix)
    ####################################################################

    def set_distributed_assembled(self, irn_loc, jcn_loc, a_loc):
        """Set the distributed assembled matrix.

        Distributed assembled matrices require setting icntl(18) != 0.
        """
        self.set_distributed_assembled_rows_cols(irn_loc, jcn_loc)
        self.set_distributed_assembled_values(a_loc)

    def set_distributed_assembled_rows_cols(self, irn_loc, jcn_loc):
        """Set the distributed assembled matrix row & column numbers.

        Distributed assembled matrices require setting icntl(18) != 0.
        """
        assert irn_loc.size == jcn_loc.size

        self._refs.update(irn_loc=irn_loc, jcn_loc=jcn_loc)
        self.id.nz_loc = irn_loc.size
        self.id.irn_loc = self.cast_array(irn_loc)
        self.id.jcn_loc = self.cast_array(jcn_loc)

    def set_distributed_assembled_values(self, a_loc):
        """Set the distributed assembled matrix values.

        Distributed assembled matrices require setting icntl(18) != 0.
        """
        assert a_loc.size == self._refs['irn_loc'].size
        self._refs.update(a_loc=a_loc)
        self.id.a_loc = self.cast_array(a_loc)


    ####################################################################
    # Right hand side entry
    ####################################################################

    def set_rhs(self, rhs):
        """Set the right hand side. This matrix will be modified in place."""
        assert rhs.size == self.id.n
        self._refs.update(rhs=rhs)
        self.id.rhs = self.cast_array(rhs)

    def set_icntl(self, idx, val):
        """Set the icntl value.

        The index should be provided as a 1-based number.
        """
        self.id.icntl[idx-1] = val

    def set_job(self, job):
        """Set the job."""
        self.id.job = job

    def set_silent(self):
        """Silence most messages."""
        self.set_icntl(1, -1) # output stream for error msgs
        self.set_icntl(2, -1) # otuput stream for diagnostic msgs
        self.set_icntl(3, -1) # output stream for global info
        self.set_icntl(4, 0)  # level of printing for errors

    @property
    def destroyed(self):
        return self.id is None

    def destroy(self):
        """Delete the MUMPS context and release all array references."""
        if self.id is not None and self._mumps_c is not None:
            self.id.job = -2 # JOB_END
            self._mumps_c(self.id)
        self.id = None
        self._refs = None

    def __del__(self):
        if not self.destroyed:
            warnings.warn("undestroyed %s" % self.__class__.__name__,
                          RuntimeWarning)
        self.destroy()

    def mumps(self):
        """Call MUMPS, checking for errors in the return code.

        The desired job should have already been set using `ctx.set_job(...)`.
        As a convenience, you may wish to call `ctx.run(job=...)` which sets
        the job and calls MUMPS.
        """
        self._mumps_c(self.id)
        if self.id.infog[0] < 0:
            raise RuntimeError("MUMPS error: %d" % self.id.infog[0])

    def run(self, job):
        """Set the job and run MUMPS.

        Valid Jobs
        ----------
        1 : Analysis
        2 : Factorization
        3 : Solve
        4 : Analysis + Factorization
        5 : Factorization + Solve
        6 : Analysis + Factorization + Solve
        """
        self.set_job(job)
        self.mumps()

class DMumpsContext(_MumpsBaseContext):

    cast_array = staticmethod(_dmumps.cast_array)
    _mumps_c = staticmethod(_dmumps.dmumps_c)
    _MUMPS_STRUC_C = staticmethod(_dmumps.DMUMPS_STRUC_C)
