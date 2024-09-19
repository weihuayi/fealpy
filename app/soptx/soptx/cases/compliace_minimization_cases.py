class ComplianceMinimizationCase:
    def __init__(self):
        self.material_properties = None
        self.geometry_properties = None
        self.filter_properties = None
        self.constraint_conditions = None
        self.boundary_conditions = None
        self.termination_criterias = None

    def initialize_material_properties(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_geometry_properties(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_filter_properties(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_constraint_conditions(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_boundary_conditions(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_termination_criteria(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_case(self):
        self.initialize_material_properties()
        self.initialize_geometry_properties()
        self.initialize_filter_properties()
        self.initialize_constraint_conditions()
        self.initialize_boundary_conditions()
        self.initialize_termination_criteria()

    def __repr__(self):
        repr_str = (f"ComplianceMinimizationCase(\n"
                    f"  material_properties = {self.material_properties},\n"
                    f"  geometry_properties = {self.geometry_properties},\n"
                    f"  constraint_conditions = {self.constraint_conditions},\n"
                    f"  filter_properties = {self.filter_properties},\n"
                    f"  boundary_conditions = {self.boundary_conditions},\n"
                    f"  termination_criterias = {self.termination_criterias}\n")
        repr_str += ")"
        return repr_str
    
class MBBBeamComplianceMinimizationCase(ComplianceMinimizationCase):
    def __init__(self, case_name: str):
        super().__init__()
        self.case_name = case_name
        self.nx = None
        self.ny = None
        self.rho = None
        self.h = None
        self.initialize_case()

    def initialize_geometry_properties(self):
        # 判断 case_name 以设置不同的几何参数
        if self.case_name == "top88":
            x_max, y_max = 6.0, 2.0
        elif self.case_name == "paperX":
            x_max, y_max = 8.0, 3.0
        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.geometry_properties = GeometryProperties(x_min=0.0, x_max=x_max, y_min=0.0, y_max=y_max)
        width, height = self.geometry_properties.get_dimensions()
        self.nx = int(width)
        self.ny = int(height)
        self.h = [width / self.nx, height / self.ny]

    def initialize_material_properties(self):
        # 根据 case_name 设置不同的材料参数
        if self.case_name == "top88":
            E0, Emin, nu, penal = 1.0, 1e-9, 0.3, 3.0
        elif self.case_name == "paperX":
            E0, Emin, nu, penal = 2.0, 1e-8, 0.35, 4.0
        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.material_properties = MaterialProperties(
            E0=E0, Emin=Emin, nu=nu, penal=penal,
            hypo="plane_stress", rho=self.rho,
            interpolation_model=SIMPInterpolation()
        )

    def initialize_filter_properties(self):
        # 根据 case_name 设置不同的滤波器参数
        if self.case_name == "top88":
            rmin, ft = 1.5, 1
        elif self.case_name == "paperX":
            rmin, ft = 2.0, 2
        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.filter_properties = FilterProperties(nx=self.nx, ny=self.ny, rmin=rmin, ft=ft)

    def initialize_constraint_conditions(self):
        # 根据 case_name 设置不同的约束条件
        if self.case_name == "top88":
            volfrac = 0.5
        elif self.case_name == "paperX":
            volfrac = 0.6
        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.constraint_conditions = ConstraintConditions()
        self.constraint_conditions.set_volume_constraint(is_on=True, vf=volfrac)
        self.volfrac = self.constraint_conditions.get_constraints()['volume']['vf']
        self.rho = self.volfrac * bm.ones(self.nx * self.ny, dtype=bm.float64)

    def initialize_boundary_conditions(self):
        # 根据 case_name 设置不同的边界条件
        if self.case_name == "top88":
            # 定义 top88 的边界条件
            def force(points: TensorLike) -> TensorLike:
                val = bm.zeros(points.shape, dtype=points.dtype)
                val[self.ny, 1] = -1
                return val

            def dirichlet(points: TensorLike) -> TensorLike:
                return bm.zeros(points.shape, dtype=points.dtype)

            def is_dirichlet_boundary_edge(edge_centers: TensorLike) -> TensorLike:
                left_edge = (edge_centers[:, 0] == 0.0)
                specific_edge = (edge_centers[:, 0] == self.nx) & (edge_centers[:, 1] == 0.5)
                result = left_edge | specific_edge
                return result

            def is_dirichlet_node() -> TensorLike:
                dirichlet_nodes = bm.zeros((self.nx + 1) * (self.ny + 1), dtype=bool)
                dirichlet_nodes[0:self.ny + 1] = True
                dirichlet_nodes[(self.ny + 1) * self.nx] = True
                return dirichlet_nodes

            def is_dirichlet_direction() -> TensorLike:
                direction_flags = bm.zeros(((self.nx + 1) * (self.ny + 1), 2), dtype=bool)
                direction_flags[0, 0] = True
                direction_flags[1, 0] = True
                direction_flags[2, 0] = True
                direction_flags[(self.ny + 1) * self.nx, 1] = True
                return direction_flags

        elif self.case_name == "paperX":
            # 定义 paperX 的边界条件
            def force(points: TensorLike) -> TensorLike:
                val = bm.zeros(points.shape, dtype=points.dtype)
                val[self.ny // 2, 1] = -2  # example force configuration
                return val

            def dirichlet(points: TensorLike) -> TensorLike:
                return bm.zeros(points.shape, dtype=points.dtype)

            def is_dirichlet_boundary_edge(edge_centers: TensorLike) -> TensorLike:
                bottom_edge = (edge_centers[:, 1] == 0.0)
                specific_edge = (edge_centers[:, 0] == self.nx / 2) & (edge_centers[:, 1] == 1.0)
                result = bottom_edge | specific_edge
                return result

            def is_dirichlet_node() -> TensorLike:
                dirichlet_nodes = bm.zeros((self.nx + 1) * (self.ny + 1), dtype=bool)
                dirichlet_nodes[self.nx:self.nx + self.ny + 1] = True
                return dirichlet_nodes

            def is_dirichlet_direction() -> TensorLike:
                direction_flags = bm.zeros(((self.nx + 1) * (self.ny + 1), 2), dtype=bool)
                direction_flags[self.nx, 0] = True
                direction_flags[self.nx + 1, 0] = True
                direction_flags[self.nx + 2, 0] = True
                direction_flags[(self.ny + 1) * self.nx // 2, 1] = True
                return direction_flags

        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.boundary_conditions = BoundaryConditions(
            force_func=force,
            dirichlet_func=dirichlet,
            is_dirichlet_boundary_edge_func=is_dirichlet_boundary_edge,
            is_dirichlet_node_func=is_dirichlet_node,
            is_dirichlet_direction_func=is_dirichlet_direction
        )

    def initialize_termination_criteria(self):
        # 根据 case_name 设置不同的终止准则
        if self.case_name == "top88":
            max_loop, tol_change = 2000, 0.01
        elif self.case_name == "paperX":
            max_loop, tol_change = 1500, 0.02
        else:
            raise ValueError(f"Unknown case name: {self.case_name}")

        self.termination_criterias = TerminationCriteria(max_loop=max_loop, tol_change=tol_change)

    def __repr__(self):
        repr_str = super().__repr__()
        repr_str = repr_str.replace('ComplianceMinimizationCase', 'MBBBeamComplianceMinimizationCase')
        repr_str = f"case_name = {self.case_name},\n" + repr_str
        return repr_str

