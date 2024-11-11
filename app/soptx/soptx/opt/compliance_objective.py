from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike as _DT

from builtins import int, str, float, object

from fealpy.typing import Union

from fealpy.opt.objective import Objective

from fealpy.mesh.mesh_base import Mesh

from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace.tensor_space import TensorFunctionSpace

from app.soptx.soptx.solver.fem_solver import FEMSolver

from app.soptx.soptx.utils.calculate_ke0 import calculate_ke0

from app.soptx.soptx.material.material_properties import ElasticMaterialProperties

from app.soptx.soptx.filter.filter_properties import FilterProperties

from app.soptx.soptx.opt.volume_objective import VolumeConstraint

from app.soptx.soptx.utils.timer import timer

class ComplianceObjective(Objective):
    """
    Compliance Minimization Objective for Topology Optimization.
    """
    def __init__(self,
                mesh: Mesh,
                space_degree: int,
                dof_per_node: int,
                dof_ordering: str,
                material_properties: ElasticMaterialProperties,
                pde: object,
                solver_method: str,
                volume_constraint: VolumeConstraint,
                filter_type: Union[int, str, None] = None,
                filter_rmin: Union[float, None] = None) -> None:
        """
        Initialize the compliance objective function.
        """
        super().__init__()
        self.mesh = mesh
        self.space_degree = space_degree
        self.dof_per_node = dof_per_node
        self.dof_ordering = dof_ordering
        self.material_properties = material_properties
        self.pde = pde
        self.solver_method = solver_method
        self.volume_constraint = volume_constraint
        self.filter_type = filter_type
        self.filter_rmin = filter_rmin

        self.space = self._create_function_space(self.space_degree, 
                                                self.dof_per_node, self.dof_ordering)
        self.filter_properties = self._create_filter_properties(self.filter_type, 
                                                            self.filter_rmin)
        self.displacement_solver = self._create_displacement_solver()

        self.ke0 = calculate_ke0(material_properties=self.material_properties, 
                                tensor_space=self.space)
        

    def _create_function_space(
        self, 
        degree: int, 
        dof_per_node: int, 
        dof_ordering: str
    ) -> TensorFunctionSpace:
        """
        Create a TensorFunctionSpace instance based on the given 
            `degree`, `dof_per_node`, and `dof_ordering`.
        """
        space_C = LagrangeFESpace(mesh=self.mesh, p=degree, ctype='C')

        if dof_ordering == 'gd-priority':
            shape = (-1, dof_per_node)
        elif dof_ordering == 'dof-priority':
            shape = (dof_per_node, -1)
        else:
            raise ValueError("Invalid `dof_ordering`. \
                                Use 'gd-priority' or 'dof-priority'.")

        return TensorFunctionSpace(scalar_space=space_C, shape=shape)

    def _create_filter_properties(
        self, 
        filter_type: Union[int, str, None], 
        filter_rmin: Union[float, None]
    ) -> Union[FilterProperties, None]:
        """
        Create a FilterProperties instance based on the given filter type and radius.
        """
        if filter_type is None:
            if filter_rmin is not None:
                raise ValueError("When `filter_type` is None, `filter_rmin` must also be None.")
            return None
        
        filter_type_mapping = {
        'sens': 0,
        'dens': 1,
        'heaviside': 2,
        }

        if isinstance(filter_type, int):
            ft = filter_type
        elif isinstance(filter_type, str):
            ft = filter_type_mapping.get(filter_type.lower())
            if ft is None:
                raise ValueError(
                    f"Invalid `filter type` '{filter_type}'. "
                    f"Please use one of {list(filter_type_mapping.keys())} or add it to the `filter_type_mapping`."
                )
        else:
            raise TypeError("`filter_type` must be an integer, string, or None.")
        
        if filter_rmin is None:
            raise ValueError("`filter_rmin` cannot be None when `filter_type` is specified.")

        return FilterProperties(mesh=self.mesh, rmin=filter_rmin, ft=ft)

    def _create_displacement_solver(self) -> FEMSolver:
        """
        Create a FEMSolver instance based on the given solver method.
        """
        return FEMSolver(material_properties=self.material_properties,
                        tensor_space=self.space,
                        pde=self.pde)

    def compute_displacement(self, rho: _DT):
        """
        Compute the displacement field 'uh' based on the density 'rho'.
        """
        material_properties = self.material_properties
        displacement_solver = self.displacement_solver

        material_properties.rho = rho

        uh = displacement_solver.solve(solver_method=self.solver_method)

        return uh
    
    def compute_ce(self, rho: _DT, uh: _DT = None) -> _DT:
        """
        Compute the element compliance values 'ce' based on the density 'rho' and displacement 'uh'.
        """
        ke0 = self.ke0

        if uh is None:
            uh = self.compute_displacement(rho)

        cell2ldof = self.space.cell_to_dof()
        uhe = uh[cell2ldof]

        ce = bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)

        return ce

    def fun(self, rho_phys: _DT) -> float:
        """
        Compute the compliance `c` based on the density `rho`.
        """
        # tmr = timer("Compliance Objective")
        # next(tmr)
        tmr = None

        uh = self.compute_displacement(rho=rho_phys)
        if tmr:
            tmr.send('Solve Displacement')

        ce = self.compute_ce(rho=rho_phys, uh=uh)
        if tmr:
            tmr.send('Compute Element Compliance')

        self.ce = ce
        self.uh = uh

        E = self.material_properties.material_model()

        c = bm.einsum('c, c -> ', E, ce) 
        if tmr:
            tmr.send('Compute Compliance Value')

        # material_properties = self.material_properties
        # displacement_solver = self.displacement_solver
        # ke0 = self.ke0

        # # `material_properties.rho` must be the physical density `rho_phys``
        # material_properties.rho = rho
        # if tmr:
        #     tmr.send('Assign Density')

        # uh = displacement_solver.solve(solver_method=self.solver_method)
        # if tmr:
        #     tmr.send('Solve Displacement')

        # cell2ldof = self.space.cell_to_dof()
        # uhe = uh[cell2ldof]
        # if tmr:
        #     tmr.send('Extract Element Displacements')

        # ce = bm.einsum('ci, cik, ck -> c', uhe, ke0, uhe)
        # self.ce = ce
        # if tmr:
        #     tmr.send('Compute Element Compliance')

        # E = self.material_properties.material_model()
        # if tmr:
        #     tmr.send('Compute Material Model')

        # c = bm.einsum('c, c -> ', E, ce)
        # if tmr:
        #     tmr.send('Compute Compliance Value')

        if tmr:
            tmr.send(None)
        
        return c
    
    def jac(self, rho: _DT, beta: float = None, rho_tilde: _DT = None) -> _DT:
        """
        Compute the gradient of compliance w.r.t. density.
        """
        ce = getattr(self, 'ce', None)
        uh = getattr(self, 'uh', None)

        if ce is None or uh is None:
            uh = self.compute_displacement(rho)
            ce = self.compute_ce(rho, uh)

        material_properties = self.material_properties
        dE = material_properties.material_model_derivative()
        dce = - bm.einsum('c, c -> c', dE, ce)

        # material_properties = self.material_properties
        # ce = self.ce

        # dE = material_properties.material_model_derivative()
        # dce = - bm.einsum('c, c -> c', dE, ce)

        if self.filter_properties is None:
            return dce

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs
        cell_measure = self.mesh.entity_measure('cell')

        if ft == 0:
            rho_dce = bm.einsum('c, c -> c', rho[:], dce)
            filtered_dce = H.matmul(rho_dce)
            dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho[:])
        elif ft == 1:
            # first normalize, then apply weight factor
            dce[:] = H.matmul(dce * cell_measure / H.matmul(cell_measure))
        elif ft == 2:
            if beta is None or rho_tilde is None:
                raise ValueError("Heaviside projection filter requires both beta and rho_tilde.")
            dxe = beta * bm.exp(-beta * rho_tilde) + bm.exp(-beta)
            dce[:] = H.matmul(dce * dxe * cell_measure / H.matmul(cell_measure))

        return dce


    def hess(self, rho: _DT, lambda_: dict) -> _DT:
        material_properties = self.material_properties

        material_properties.rho = rho
        E = material_properties.material_model()
        dE = material_properties.material_model_derivative()

        ft = self.filter_properties.ft
        H = self.filter_properties.H
        Hs = self.filter_properties.Hs

        _, ce, _ = self._compute_uhe_and_ce(rho)
        coef = 2 * dE**2 / E
        hessf = coef.matmul(ce)

        if ft == 0:
            hessf = hessf
        elif ft == 1:
            hessf[:] = H.matmul(hessf) / Hs

        hessc = self._compute_constraint_hessian(rho)

        if 'ineq' in lambda_:
            hessian = bm.diag(hessf) + lambda_['ineq'] * hessc
        else:
            hessian = bm.diag(hessf)
        
        return hessian
    
    def _compute_constraint_hessian(self, rho: _DT) -> _DT:
        """
        Compute the Hessian of the constraint function for non-linear constraints.

        Parameters:
            rho (_DT): Current design variables (density distribution).

        Returns:
            _DT: Hessian matrix of the constraint function.
        """
        # For linear constraints, this would be zero
        # For non-linear constraints, provide the Hessian computation here
        return 0  # Placeholder for linear constraint case
    


