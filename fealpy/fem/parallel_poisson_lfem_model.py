from mpi4py import MPI

from ..backend import backend_manager as bm
from ..mesh.parallel import ParallelMesh, split_homogeneous_mesh
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC
from ..functionspace import LagrangeFESpace
from fealpy.solver import spsolve


class ParallelPoissonLFEMModel:
    def __init__(self, mesh, pde, divi_line, max_iter=1000, tol=1e-6):
        """
        Parallel computing of the two-dimensional and three-dimensional Poisson equations

        Parameters:
            mesh: Triangular meshes or quadrilateral meshes
            pde: Poisson equation
            divi_line: Dividing line
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
        """
        self.COMM = MPI.COMM_WORLD
        self.RANK = self.COMM.Get_rank()
        self.SIZE = self.COMM.Get_size()
        
        self.mesh = mesh
        self.pde = pde
        self.divi_line = divi_line
        self.max_iter = max_iter
        self.tol = tol
        self.kwargs = {"device":mesh.device,"dtype":bm.float64}

    def region_division(self):
        """
       Perform mesh partitioning
        """
        barycenter = self.mesh.entity_barycenter("cell")
        K = bm.arange(self.mesh.number_of_cells())

        first_cell_number = K[(barycenter[:,0] < self.divi_line[1])]
        second_cell_number = K[(barycenter[:,0] > self.divi_line[0])]

        return first_cell_number, second_cell_number
    
    def solve_subdomain(self, pmesh, uh0, space, a, b):
        """
        Solve the sub-region problem

        Parameters:
            pmesh: The partitioned sub - meshes
            uh0: The initial solution
            space: The function space
            a:The degrees of freedom of the artificial boundary of the sub - mesh
            b:The degree-of-freedom numbers of the sub - mesh among those of other meshes
        """      
        NN = pmesh.number_of_nodes()
        node = pmesh.entity("node")
        uh1 = bm.zeros(NN, **self.kwargs)

        beform = BilinearForm(space)
        beform.add_integrator(ScalarDiffusionIntegrator(coef=1, q=1+3))
        A = beform.assembly()

        leform = LinearForm(space)
        leform.add_integrator(ScalarSourceIntegrator(self.pde.source))
        F = leform.assembly()

        isBdNode = pmesh.boundary_node_index()
        uh1[isBdNode] =self.pde.dirichlet((node[isBdNode])).reshape(-1)
        uh1[a] = uh0[b]
        A, F = DirichletBC(space, gd=uh1).apply(A, F) 

        uh = space.function()
        uh[:] = spsolve(A, F, "scipy")

        return uh
    
    def solve_schwarz_alternating(self, uh0):
        """
        Solve the equations using the Schwarz alternating method.

        Parameters:
            uh0: The initial solution
        """   
        if self.RANK == 0:
            first_cell_number, second_cell_number = self.region_division()
            ranges = []
            ranges.append(first_cell_number)
            ranges.append(second_cell_number)

            data_list = list(split_homogeneous_mesh(self.mesh, masks=ranges))
        else:
            data_list = [None,] * self.SIZE

        data = self.COMM.scatter(data_list, root=0) 
        del data_list

        pmesh = ParallelMesh(self.RANK, *data) 

        a1 = ~pmesh.global_flag_on_boundary("node")
        a2 = pmesh.boundary_node_index()
        a = a2[a1]

        b1 = pmesh.global_indices("node")
        b = b1[a]
   
        space = LagrangeFESpace(self.mesh, p=1)
        space1 = LagrangeFESpace(pmesh, p=1)
    
        iter_num = 0
        while iter_num < self.max_iter:
            uh = self.solve_subdomain(pmesh, uh0, space1, a, b)
            Eh1 = bm.copy(uh0)

            k = pmesh.global_indices("node")
            Eh1[k] = uh

            if self.RANK == 0:
                Eh = space.function()
            else:
                Eh = None
      
            self.COMM.Reduce(Eh1, Eh, op=MPI.SUM, root=0)

            if self.RANK == 0:
                Eh = 0.5 * Eh
                print(f"第{iter_num+1}次迭代，总误差为{self.mesh.error(self.pde.solution, Eh.value)}")
            
            data1 = self.COMM.bcast(Eh, root=0)
            if bm.max(bm.abs(data1-uh0))<self.tol:
                break
            uh0 = data1
            iter_num += 1

        return uh0