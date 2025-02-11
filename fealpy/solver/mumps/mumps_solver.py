
from .dmumps_context import DMumpsContext

_solvers ={
        "D": DMumpsContext,
        }

class MumpsSolver:
    def __init__(self, stype='D', 
                 par=1, sym=0, comm=None, 
                 silent=True, timer=None, logger=None):
        """Initialize the MumpsSolver object.

        Note
            1. COMMCOMM (integer) must be set by the user on all processors 
               before the initialization phase (JOB= –1) and
               must not be changed in further calls 
            2. PAR (integer) must be initialized by the user on all processors 
               before the initialization phase (JOB= –1) and is accessed by 
               MUMPS only during this phase. 
               0: the host is not involved in the parallel steps of the factorization and solve phases.
               1: the host is also involved in the parallel steps of the factorization and solve phases.
            3. SYM (integer) must be initialized by the user on all processors 
               before the initialization phase (JOB= –1) and is accessed by 
               MUMPS only during this phase. 
               0: unsym; 1: spd; 2 symmetric
            4. ICNTL(5) controls the matrix input format. 
               0: assembled format (centralized on the host or distributed)
               1: elemental format
            5. ICNTL(18): controls the options to distribute the matrix
            6. ICNTL(9): compute the solution using A or A^T.
            7. ICNTL(16): ontrols the setting of the number of OpenMP threads by 
               MUMPS and should be used only when the setting of multithreaded 
               parallelism is not possible outside MUMPS

        """
        self.timer = timer
        self.logger = logger
        self.context = _solvers[stype](par=par, sym=sym, comm=comm)
        if silent:
            self.context.set_silent()

    def solve(self, A, b):
        """
        """
        x = b.copy()
        self.context.set_centralized_sparse(A)
        self.context.set_rhs(x)

        self.context.run(6)

        return x 

    def solve_triangular(self, A, b, transpose=False):
        """Solve triangular system Ax=b or A^Tx=b 

        Parameters:
            A (CSRTensor): the upper or lower triangular matrix 
            b (TensorLike): the right hand side
            transpose (bool): the flag for 
        """
        #self.context.set_icntl(16, 10)
        if transpose: # ICNTL(9) compute the solution using A or A^T
            self.context.set_icntl(9, 2) # solve A^T x = b
        return self.solve(A, b)
