import torch.nn as nn

from typing import Union, Optional

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.poisson import PoissonPDEDataT
from fealpy.ml.modules import PoUSpace, PoUSin, RandomFeatureSpace
from fealpy.ml.modules import Sin, Cos, Tanh

activations = {"Sin": Sin(), "Cos": Cos(), "Tanh": Tanh()}

class PoissonRFMModel(ComputationalModel):
    """
    A computational model for solving Poisson's equation using Random Feature Methods (RFM).

    This model implements various RFM approaches including global random feature space,
    partition of unity (PoU) space, and a combination of both to solve Poisson's equation
    with dirichlet boundary conditions. It supports multiple activation functions and
    provides comprehensive visualization capabilities.

    Parameters:
        options(dict): Configuration dictionary containing model parameters. If None, default parameters
            from get_options('Tanh', 'Sin', 'Cos') will be used. Key parameters include:
            - pde(int or PoissonPDEDataT): PDE definition;
            - rfm_type(str): Type of RFM approach, can choose from 'global', 'pou', or 'both';
            - mesh_cen_size(Union[int, tuple]): Mesh size for center points generation;
            - mesh_col_size(Union[int, tuple]): Mesh size for collocation points generation;
            - mesh_error_size(Union[int, tuple]): Mesh size for error computation;
            - nbasis(int): Number of basis functions;
            - uniform(tuple): Uniform distribution parameters for random feature weights;
            - activation(str): Activation function type, can choose from 'Tanh', 'Sin', or 'Cos';
            - pbar_log(bool): Progress bar logging flag;
            - log_level(str): Logging level, can choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

    Attributes:
        pde(PoissonPDEDataT): The Poisson PDE problem definition.

        options(dict): Configuration parameters for the model.

        pbar_log(bool): Flag indicating whether to use progress bar for logging.

        log_level(str): Logging level. Logging level, can choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        mesh_cen_size(Union[int, tuple]): Mesh size for center points generation.

        mesh_col_size(Union[int, tuple]): Mesh size for collocation points generation.

        nbasis(int): Number of basis functions in each partition or globally.

        uniform(tuple): Uniform distribution parameters for random feature weights.

        activation(nn.Module): Activation function module.

        rfm_type(str): Type of RFM approach. Type of RFM approach ('global', 'pou', or 'both').

        tmr(timer): Timer instance for performance measurement.

        space(Union[RandomFeatureSpace, PoUSpace, tuple]): Feature space used for approximation.

        solution(Union[Function, tuple]): Computed solution function.

    Methods:
        get_options(): Get default configuration parameters.

        set_pde(): Initialize the PDE problem definition.

        set_mesh(): Create computational mesh.

        globalspace(): Create global random feature space.

        pouspace():  Create partition of unity space.

        bothspace():  Create combined global and PoU spaces.

        get_collocation_points():  Generate collocation points for training.

        linear_system(): Assemble the linear system for solving.

        predict():  Make predictions at given points.

        run(): Execute the complete solution process.

        show(s=80):  Visualize results and compute errors.

    Examples:
        >>> from fealpy.backend import bm
        >>> bm.set_backend('pytorch')
        >>> from fealpy.ml import PoissonRFMModel  
        >>> options = PoissonRFMModel.get_options()    # Get the default options of the network  
        >>> model = PoissonRFMModel(options=options)  # Initialize the model with the default options  
        >>> model.run()                                # Run the model  
        >>> model.show()
    """
    def __init__(self, options:dict={}):
        self.options = self.get_options()
        self.options.update(options)
        
        self.pbar_log = self.options['pbar_log']
        self.log_level = self.options['log_level']
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)

        self.mesh_cen_size = self.options['mesh_cen_size']
        self.mesh_col_size = self.options['mesh_col_size']
        self.nbasis = self.options['nbasis']
        self.uniform = self.options['uniform']
        self.activation = activations[self.options['activation']]
        self.rfm_type = self.options['rfm_type']
        self.tmr = timer() 

        self.set_pde(self.options['pde'])

    @classmethod
    def get_options(cls):
        """Get default configuration parameters for the model.

        Returns:
            options(dict): Dictionary containing all configuration parameters with parameter names as keys and default values.
        """

        import argparse
        parser = argparse.ArgumentParser(description=
                "Poisson equation solver using Random Feature Methods.")

        parser.add_argument('--pde',default=10, type=int,
                            help="Built-in PDE example ID for different Poisson problems, default is 10.")
        
        parser.add_argument('--rfm_type', default="pou", type=str,
                            help="Random feature function type, options are 'global', 'pou', 'both'.")
        
        parser.add_argument('--mesh_cen_size', default=3, type=Union[int, tuple],
                            help='Number of grid points along each dimension, used to generate the center \
                                points of each partition, default is 3.')
        
        parser.add_argument('--mesh_col_size', default=100, type=Union[int, tuple],
                            help='Number of grid points along each dimension, used to generate the \
                                collocation points, default is 100.')
        
        parser.add_argument('--mesh_error_size', default=80, type=Union[int, tuple],
                            help='Number of grid points along each dimension, used to compute the \
                                error, it is best not to set a very large value, default is 80.')

        parser.add_argument('--nbasis', default=160, type=int,
                            help='Number of basis functions in each partition or globally, default is 160.')
        
        parser.add_argument('--uniform', default=(1, bm.pi), type=tuple,
                            help='Parameters: k(weight), b(bias) in random feature function are \
                                generated from uniform distribution, default is (1, pi).')

        parser.add_argument('--activation', default="Cos", type=str,
                            help="Activation function in random feature function, default is Tanh, \
                            options are 'Tanh', 'Sin', 'Cos'.")
        
        parser.add_argument('--pbar_log', default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level', default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        """Initialize the PDE problem definition.
        
        Parameters:
            pde(Union[PoissonPDEDataT, int]): Either a Poisson's equation problem object or the ID (integer) of a predefined example. 
                If an integer, the corresponding predefined Poisson's equation problem is retrieved from the PDE model manager.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde 

    def set_mesh(self, mesh_size: Union[int, tuple]):
        """Create computational mesh.

        Parameters:
            mesh_size(tuple of int Number of nodes in each dimension.

        Returns:
            mesh: The created mesh object.
        """
        pde = self.pde
        gd = pde.geo_dimension()
        if isinstance(mesh_size, int):
            self.mesh_size = (mesh_size, ) * gd
        elif isinstance(mesh_size, tuple):
            n = len(mesh_size)
            if n == 1:
                self.mesh_size = mesh_size * gd
            else:
                assert n == gd, f"Dimension of mesh size {n} does not match dimension of problem {gd}."
        cell_size = tuple(x - 1 for x in self.mesh_size)
        mesh = self.pde.init_mesh['uniform'](*cell_size)
        return mesh

    def globalspace(self):
        """
        Create a global random feature space.

        Returns:
            RandomFeatureSpace: Global random feature space instance.
        """
        gd = self.pde.geo_dimension()
        act = self.activation
        bound = self.uniform
        Jn = self.nbasis
        space = RandomFeatureSpace(in_dim=gd, nf=Jn, activate=act, bound=bound)
        return space

    def pouspace(self):
        """Create a partition of unity (PoU) space.

        Returns:
            PoUSpace: Partition of unity space instance.
        """
        factory = self.globalspace()
        mesh = self.set_mesh(self.mesh_cen_size)
        node = mesh.entity('node')
        h = mesh.h / 2
        space = PoUSpace(factory, pou=PoUSin(), centers=node, radius=h, print_status=False)
        del mesh
        return space
    
    def bothspace(self):
        """
        Create both global and PoU spaces.

        Returns:
            tuple: Tuple containing (globalspace, pouspace).
        """
        space = (self.globalspace(), self.pouspace())
        return space
    
    def get_collocation_points(self):
        """
        Generate collocation points for training.

        Returns:
            tuple: Tuple containing (interior_points, boundary_points) tensors.
        """
        mesh = self.set_mesh(self.mesh_col_size)
        _bd_node = mesh.boundary_node_flag()
        col_in = mesh.entity('node', index=~_bd_node)
        col_bd = mesh.entity('node', index=_bd_node)
        del _bd_node, mesh
        self.tmr.send(f"Collocation points assembly time")
    
        return col_in, col_bd
    
    def linear_system(self):
        """
        Assemble the linear system for solving the Poisson equation.

        Returns:
            tuple: Tuple containing (A, b) where A is the system matrix and b is the right-hand side vector.
        """
        from fealpy.sparse import csr_matrix

        if self.rfm_type == 'global':
            self.space = self.globalspace()
            self.logger.info(f"\nGlobal space with {self.space.nf} basis functions.")
            self.tmr.send(f"Global space assembly time")

        elif self.rfm_type == 'pou':
            self.space = self.pouspace()
            self.logger.info(f"\nPoU space with {self.space.number_of_partitions()} partitions"
                          f" and {self.space.number_of_basis()} basis functions.")
            self.tmr.send(f"PoU space assembly time")

        elif self.rfm_type == 'both':
            self.space = self.bothspace()
            self.logger.info(f"\nBoth space with {self.space[0].number_of_basis()} global basis functions"
                          f" and {self.space[1].number_of_basis()} PoU basis functions in"
                          f" {self.space[1].number_of_partitions()} partitions.")
            self.tmr.send(f"Both space assembly time")

        else:
            raise ValueError(f"Error: Illegal rfm_type {self.rfm_type}")

        col_in, col_bd = self.get_collocation_points()

        pde = self.pde
        b = bm.concat([pde.source(col_in), pde.dirichlet(col_bd)], axis=0)
        if self.rfm_type == 'both':
            A1 = self.space[0].laplace_basis(col_in)
            A2 = self.space[1].laplace_basis(col_in)
            laplace_phi = bm.concat([A1, A2], axis=-1) 
            A1 = self.space[0].basis(col_bd)
            A2 = self.space[1].basis(col_bd)
            phi = bm.concat([A1, A2], axis=-1)
            del A1, A2 
        else:
            laplace_phi = self.space.laplace_basis(col_in)
            phi = self.space.basis(col_bd)

        del col_in, col_bd
        A = bm.concat([-laplace_phi, phi], axis=0)
        del laplace_phi, phi

        b = A.T @ b
        A = A.T @ A
        A = csr_matrix(A)
        self.tmr.send(f"Linear system assembly time")
        return A, b
    
    def predict(self, p: TensorLike) -> TensorLike:
        """
        Make predictions at given points.

        Parameters:
            p(TensorLike):  Points at which to evaluate the solution.

        Returns:
            TensorLike: Predicted values at the given points.
        """
        if self.rfm_type == 'both':
            f1 = self.space[0].function(self.solution[0])
            f2 = self.space[1].function(self.solution[1])
            return f1(p) + f2(p)
        else:
            return self.solution(p)
    
    def run(self):
        """
        Execute the complete solution process including linear system assembly and solving.
        """
        next(self.tmr)
        from fealpy.solver import spsolve
        A, b = self.linear_system()
        um = spsolve(A, b, solver='scipy')
        del A, b
        if self.rfm_type == 'both':
            f1 = self.space[0].function(um[:self.nbasis])
            f2 = self.space[1].function(um[self.nbasis:])
            self.solution = (f1, f2)
        else:
            self.solution = self.space.function(um)
        self.tmr.send(f"Solution assembly time")
        next(self.tmr)

    def show(self):
        """
        Visualize results and compute errors.
        """
        pde = self.pde
        gd = pde.geo_dimension()
        domain = pde.domain()
        s = self.options['mesh_error_size']
        if isinstance(s, int):
            s = (s, ) * gd
        elif isinstance(s, tuple):
            n = len(s)
            if n == 1:
                s = s * gd
            else:
                assert n == gd, f"Dimension of mesh size {n} does not match dimension of problem {gd}."
        mesh_err = pde.init_mesh(*s)  # TriangleMesh

        if self.rfm_type == 'both':
            error = self.solution[0].estimate_error_tensor(pde.solution, mesh=mesh_err, solution2=self.solution[1])
        else:
            error = self.solution.estimate_error_tensor(pde.solution, mesh=mesh_err)
        self.logger.info(f"L-2 error in mesh of size (nx,ny) =  {s} is: {error.item()}")

        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(10, 10))

        if gd == 2 : 
            axes = fig.add_subplot(131)
            if self.rfm_type == 'both':
                qm = self.solution[0].diff(pde.solution, solution2=self.solution[1]).add_pcolor(axes, box=domain, nums=list(s))
            else:
                qm = self.solution.diff(pde.solution)
                qm = qm.add_pcolor(axes, box=domain, nums=list(s))
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            axes.set_aspect('equal')
            axes.title.set_text('Error in RFM Solution and Exact Solution')
            fig.colorbar(qm)

            axes = fig.add_subplot(132, projection='3d')
            node = mesh_err.entity('node') 
            if self.rfm_type == "both": 
                pred = self.solution[0](node).flatten() + self.solution[1](node).flatten()
                surf2 = axes.plot_trisurf(node[:, 0], node[:, 1],
                                     pred, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
                fig.colorbar(surf2, ax=axes, shrink=0.5, label='value')
            else:
                self.solution.add_surface(axes, box=domain, nums=list(s))
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            axes.set_zlabel('u')
            axes.title.set_text('RFM Solution')   

            axes = fig.add_subplot(133, projection='3d')
            tu = self.pde.solution(node).flatten()
            surf2 = axes.plot_trisurf(node[:, 0], node[:, 1],
                                    tu, cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.8)
            fig.colorbar(surf2, ax=axes, shrink=0.5, label='value')
            axes.set_xlabel('x')
            axes.set_ylabel('y')
            axes.set_zlabel('u')
            axes.title.set_text('Exact Solution')

            plt.suptitle('Comparison RFM Solution and Exact Solution')

        elif gd == 1:
            node = mesh_err.entity('node')
            pre = bm.to_numpy(self.solution(node)).flatten()
            ture = bm.to_numpy(pde.solution(node))
            axes = fig.add_subplot()
            axes.plot(node, pre, 'b-', linewidth=3, label='RFM Solution')
            axes.plot(node, ture, 'g-', linewidth=2, label='Exact Solution')
            axes.plot(node, pre - ture, 'r-', linewidth=1, label='Error')
            axes.set_xlabel('x')
            axes.set_ylabel('u')
            axes.title.set_text('RFM Solution and Exact Solution')
            plt.legend(fontsize=12)
            plt.grid(True, linestyle=':')
            plt.suptitle('Comparison RFM Solution and Exact Solution')
        plt.show()

