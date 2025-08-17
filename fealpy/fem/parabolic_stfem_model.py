from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEModelManager, ComputationalModel
from ..model.parabolic import ParabolicPDEDataProtocol
from ..decorator import variantmethod,barycentric
from ..mesh import Mesh
# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import DirichletBC
from ..fem import SpaceTimeConvectionIntegrator
from ..fem import SpaceTimeSourceIntegrator
from ..fem import SpaceTimeDiffusionIntegrator
from ..fem import SpaceTimeMassIntegrator


class ParabolicSTFEMModel(ComputationalModel):
    """
    A class for solving parabolic PDEs using space-time finite element methods (STFEM).
    This class extends the PDEModelManager to handle parabolic problems with time-dependent solutions.
    The equation is typically of the form:
        u_t - div(K(x)∇u) + beta * ∇u + gamma*u = f
    where K is the diffusion coefficient, beta is the convection coefficient, 
    gamma is the reaction term, and f is the source term.
    
    We will transform this into a space-time problem:
        - div(D(x)∇_y(u)) + b * ∇_y(u) + gamma * u = f
    where y = (x, t) is the space-time variable 
    and D(x) is the diffusion tensor [[K(x),0],[0,0]]. b is the convection vector [beta, 1].
    """
    def __init__(self, options: Optional[dict] = None):
        super().__init__(pbar_log=options['pbar_log'], 
                         log_level=options['log_level'])
        self.options = options
        self.pdm = PDEModelManager("parabolic")
        
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], **options['mesh_size'])
        self.set_space_degree(options['space_degree'])
        self.set_quadrature(options['quadrature'])
        self.set_assemble_method(options['assemble_method'])
        self.solve.set(options['solver'])
        self.set_function_space()
        
    def set_pde(self,pde: Union[ParabolicPDEDataProtocol,int] = 1):
        """
        Set the PDE data for the model.
        
        Parameters:
            pde (str or ParabolicPDEDataProtocol): The name of the PDE or an instance of ParabolicPDEDataProtocol.
        """
        if isinstance(pde, int):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self,mesh: Union[Mesh,str] = 'uniform_tri', **kwargs):
        """
        Set the initial mesh for the model. 
        Parameters:
            mesh (Mesh or str): The mesh object or a string identifier for the mesh type.
            **kwargs: Additional keyword arguments for mesh creation.
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
            
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.") 
    
    def set_space_degree(self, p: int = 1):
        """
        Set the order of the Lagrange finite element space.
        
        Parameters:
            order (int): The polynomial order of the finite element space.
        """
        self.p = p
        self.logger.info(f"Finite element space set with polynomial order {self.p}.")
    
    def set_quadrature(self, q: int = 4):
        """
        Set the index of quadrature formula for numerical integration.
        
        Parameters:
            q (int): The order of the quadrature rule.
        """
        self.q = q
        qf = self.mesh.quadrature_formula(q)
        self.bcs,self.ws = qf.get_quadrature_points_and_weights()
        self.logger.info(f"Quadrature order set to {self.q}.")
        
    def set_assemble_method(self, method: Union[str,None] = None):
        """
        Set the method for assembling the matrix.
        
        Parameters:
            method (str): The assembly method to use ('sparse' for sparse matrix assembly).
        """
        self.assemble_method = method
        self.logger.info(f"Assembly method set to {self.assemble_method}.")
        
    def set_function_space(self):
        """
        Set the function space for the finite element method.
        This method initializes the Lagrange finite element space based on the mesh and polynomial order.
        """
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.logger.info(f"Function space initialized with polynomial order {self.p}.")
        self.uh = self.space.function()
        
    def linear_system(self):
        """
        Set up the linear system for the parabolic PDE.
        This method initializes the bilinear and linear forms, and sets up the Dirichlet boundary conditions.
        """
        space = self.space
        pde = self.pde
        method = self.assemble_method
        q = self.q
        self.bform = BilinearForm(space)
        self.lform = LinearForm(space)
        self.logger.info("Linear system set with bilinear and linear forms.")
        STMI = SpaceTimeMassIntegrator(coef=pde.reaction_coef, q=q, method=method,conv_coef=pde.convection_coef)
        STDI = SpaceTimeDiffusionIntegrator(coef=pde.diffusion_coef, q=q, method=method, conv_coef=pde.convection_coef)
        STCI = SpaceTimeConvectionIntegrator(coef=pde.convection_coef, q=q, method=method)
        STSI = SpaceTimeSourceIntegrator(source=pde.source, q=q, method=method, conv_coef=pde.convection_coef)

        self.logger.info("Integrators for mass, diffusion, convection, and source terms created.")
        self.bform.add_integrator(STMI, STDI, STCI)
        self.logger.info("Bilinear form set with diffusion, convection and mass terms.")
        self.lform.add_integrator(STSI)
        self.logger.info("Linear form set with source term.")
        
        A = self.bform.assembly()
        self.logger.info(f"Linear system initialized with {A.shape} matrix.")
        b = self.lform.assembly()
        self.logger.info(f"Linear system initialized with {b.shape} right-hand side vector.")
        self.logger.info("Bilinear and linear forms assembled.")
        
        return A, b
    
    def apply_bc(self,A,b):
        """
        Apply Dirichlet boundary conditions to the linear system.
        This method sets up the Dirichlet boundary conditions based on the PDE data.
        """
        space = self.space
        pde = self.pde
        threshold_space = pde.is_dirichlet_boundary
        threshold_time = pde.is_init_boundary
        gd_space = pde.dirichlet
        gd_time = pde.init_solution

        bc_space = DirichletBC(space=space, gd=gd_space, threshold=threshold_space)
        bc_time = DirichletBC(space=space, gd=gd_time, threshold=threshold_time)
        A, b = bc_space.apply(A, b)
        A, b = bc_time.apply(A, b)
        self.logger.info("Dirichlet boundary conditions applied with space and time.")
        return A, b
    
    @variantmethod("direct")
    def solve(self, A, b):
        """
        Solve the linear system Ax = b.
        
        Parameters:
            A (TensorLike): The coefficient matrix of the linear system.
            b (TensorLike): The right-hand side vector.
        
        Returns:
            TensorLike: The solution vector u.
        """
        from ..solver import spsolve
        uh = spsolve(A, b, solver='scipy')
        self.logger.info("Linear system solved.")
        return uh

    @solve.register("cg")
    def solve(self,A,b):
        """
        Solve the linear system using the conjugate gradient method.
        Parameters:
            A (TensorLike): The coefficient matrix of the linear system.
            b (TensorLike): The right-hand side vector.
        Returns:
            TensorLike: The solution vector u.
        """
        from ..solver import cg
        uh, info = cg(A, b, maxit=10000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return uh
    
    @solve.register("lgmres")
    def solve(self,A,b):
        from ..solver import lgmres
        uh, info = lgmres(A, b, maxit=10000, atol=1e-14, rtol=1e-14)
        res = info['residual']
        res_0 = bm.linalg.norm(b)
        stop_res = res/res_0
        self.logger.info(f"LGMRES solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return uh

    @variantmethod
    def run(self,show_error=True):
        A, b = self.linear_system()
        A, b = self.apply_bc(A, b)
        self.uh[:] = self.solve(A, b)
        self.logger.info("Run completed.")
        if show_error:
            l2 , h1 = self.error()
            self.logger.info(f"L2 Error: {l2}, H1 Error: {h1}.")

    @run.register("refine")
    def run(self,maxit=4,plot_error=False):
        """
        Run the model with uniform mesh refinement.
        Parameters:
            maxit (int): The maximum number of refinement iterations.
        """
        error_matrix = bm.zeros((maxit+1,2), dtype=bm.float64)
        for i in range(maxit):
            A, b = self.linear_system()
            A, b = self.apply_bc(A, b)
            self.uh[:] = self.solve(A, b)
            l2,h1 = self.error()
            self.logger.info(f"{i}-th step with  L2 Error: {l2}, H1 Error: {h1}.")
            error_matrix[i,0] = l2
            error_matrix[i,1] = h1
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                ms = self.options['mesh_size']
                if 'nz' in ms:              # 3D (or has z direction)
                    ms['ny'] *= 2
                    ms['nx'] *= 2
                    ms['nz'] = int(ms['nx']**2*1.2)
                else:
                    ms['nx'] *= 2
                    ms['ny'] = int(ms['nx']**2*1.2)
                self.mesh = self.pde.init_mesh[self.options['init_mesh']](**ms)
                self.space = LagrangeFESpace(self.mesh, p=self.p)
                self.uh = self.space.function()
        if plot_error:
            self.show_order(error_matrix,maxit)

    def error(self):
        gradient = lambda p : self.pde.gradient(p)[...,:-1]
        @barycentric
        def grad_value(bcs):
            return self.uh.grad_value(bcs)[...,:-1]
        
        l2 = self.mesh.error(self.pde.solution, self.uh, q = self.q)
        h1 = self.mesh.error(gradient, grad_value, q = self.q)
        return l2, h1

    def show_solution(self):
        """
        Visualize the solution using the mesh's plotting capabilities.
        """
        import matplotlib.pyplot as plt
        from fealpy.mmesh.tool import linear_surfploter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        linear_surfploter(ax , self.mesh, self.uh)
        plt.show()
        self.logger.info("Solution visualized.")
        
    def show_order(self, error_matrix,maxit):
        """
        Show the convergence order of the solution.
        This method plots the convergence order based on the error matrix.
        """
        import matplotlib.pyplot as plt
        h0 = (bm.min(self.mesh.entity_measure('cell')))**(1/self.mesh.TD)
        h = bm.array([h0 / (2 ** i) for i in range(maxit + 1)])
        log2h = bm.log2(h)

        l2_error = error_matrix[:, 0]
        h1_error = error_matrix[:, 1]
        print("L2_order:", (bm.log2(l2_error[1:] / l2_error[:-1])/ bm.log2(h[1:] / h[:-1]))[:-1])
        print("H1_order:", (bm.log2(h1_error[1:] / h1_error[:-1])/ bm.log2(h[1:] / h[:-1]))[:-1])
        plt.figure()
        # 横坐标用 -log2(h)，让曲线从左到右下降
        x = -log2h

        plt.plot(x, bm.log2(l2_error), 'o-', label='L2 error')
        plt.plot(x, bm.log2(h1_error), '^-', label='H1 error')

        # 理论参考线
        ref2 = (self.p+1) * (-x + x[0]) + 0.9*bm.log2(l2_error[0])
        plt.plot(x, ref2, 'k--', label=f'{self.p+1}nd order (ref)')

        ref1 = self.p * (-x + x[0]) + 0.9*bm.log2(h1_error[0])
        plt.plot(x, ref1, 'k-.', label=f'{self.p}nd order (ref)')

        plt.xlabel('|log2(h)|')
        plt.ylabel('log2(error)')
        plt.title('Convergence rate')
        plt.legend()
        plt.grid(True)
        # plt.gca().set_aspect('equal')
        plt.show()

    def slicing_error(self, t: float=None, p: float=None, *, tol: float=1e-10,
                      return_error: bool=False):
        """
        Compute L2 / H1 errors along a 1D slice of the space-time mesh (either fixing time t to
        obtain a spatial line, or fixing spatial coordinate x = p to obtain a temporal line),
        for arbitrary polynomial degree p of space‑time Lagrange elements.

        Parameters:
            t : float, optional
                Fixed time value. If provided, a spatial line (varying x) at time t is extracted.
            p : float, optional
                Fixed spatial x value. If provided, a temporal line (varying t) at x = p is extracted.
            tol : float, default 1e-10
                Tolerance used to identify nodes lying on the slice.
            return_error : bool, default False
                If True, return a dictionary of errors; otherwise only log the results.
        """
        from fealpy.mesh import IntervalMesh

        node = self.mesh.node                  # (NN,2) 约定: [:,0]=x, [:,1]=t
        edge = self.mesh.edge                  # (NE,2)
        results = {}

        def _build_slice(fixed_val: float, fixed_axis: int, label: str):
            """
            fixed_axis : 1 表示固定 t, 沿 x 方向; 0 表示固定 x, 沿 t 方向.
            label      : 't' 或 'x'
            返回 (IntervalMesh, LagrangeFESpace, 1D 解函数 uh1d)
            """
            # 1. 找截线节点
            idx = bm.where(bm.abs(node[:, fixed_axis] - fixed_val) < tol)[0]
            # 2. 找截线完整边
            mask_edge = bm.all(bm.isin(edge, idx), axis=1)
            slice_edge_ids = bm.where(mask_edge)[0]
            # 3. 取边并排序 (按可变轴坐标最小值)
            var_axis = 1 - fixed_axis  # 截线方向坐标轴
            e_nodes = edge[slice_edge_ids]                  # (E,2)
            ncoord = node[:, var_axis]
            key = bm.minimum(ncoord[e_nodes[:,0]], ncoord[e_nodes[:,1]])
            order = bm.argsort(key)
            e_nodes = e_nodes[order]
            slice_edge_ids = slice_edge_ids[order]
            # 4. 保证边方向 (起点 -> 终点) 在 var_axis 方向单调递增
            c0 = ncoord[e_nodes[:,0]]
            c1 = ncoord[e_nodes[:,1]]
            flip = c1 < c0
            if bm.any(flip):
                # 交换需要翻转的边两个端点
                tmp = e_nodes[flip,0].copy()
                e_nodes = bm.set_at(e_nodes, (flip,0), e_nodes[flip,1])
                e_nodes = bm.set_at(e_nodes, (flip,1), tmp)

            # 5. 获取对应边的 (p+1) 个边自由度值 (全局编号)
            e2dof = self.space.edge_to_dof(slice_edge_ids)   # (E, p+1)
            # 与几何方向一致: 若翻转则反转对应 dof 行
            if bm.any(flip):
                # 需要重新按排序索引回到 flip 布尔掩码对应
                flip_sorted = flip  # 已与 e_nodes 同步
                row_flip_indices = bm.where(flip_sorted)[0]
                for r in row_flip_indices:
                    e2dof_r = e2dof[r]
                    e2dof = bm.set_at(e2dof, r, e2dof_r[::-1])
            # 6. 取得对应数值
            edge_vals = self.uh[e2dof]          # (E, p+1)
            # 7. 生成 1D 网格 (端点坐标按每条边端点串联)
            E = e2dof.shape[0]
            # 边端点沿方向坐标
            start_coord = ncoord[e_nodes[:,0]]
            end_coord   = ncoord[e_nodes[:,1]]
            # 端点序列: 第 0 条边起点 + 所有边终点
            nodes_1d = bm.concat([start_coord[:1], end_coord], axis=0)  # (E+1,)
            # 保证严格单调 (若有重复或倒序, 说明网格质量/截取有问题)
            if bm.any(nodes_1d[1:] - nodes_1d[:-1] <= 0):
                self.logger.warning(f"Non-monotone parameter nodes on {label}={fixed_val}")
            # 构造 1D IntervalMesh (几何只用端点)
            cells_1d = bm.stack([bm.arange(E), bm.arange(1, E+1)], axis=1)
            int_mesh = IntervalMesh(nodes_1d, cells_1d)
            # 8. 构造 1D p 次 Lagrange 空间
            fe_space = LagrangeFESpace(int_mesh, p=self.p)
            cell2dof_1d = fe_space.cell_to_dof()      # (E, p+1)
            # 9. 原 2D 边 DOF 值展平为 (E*p + 1,) 与 1D 全局 DOF 一一对应:
            flat_list = [edge_vals[0]]
            if E > 1:
                flat_list.extend(edge_vals[i,1:] for i in range(1, E))
            flat_vals = bm.concat(flat_list, axis=0)  # 形状 (E*p + 1,)
            ndof_1d = fe_space.number_of_global_dofs()
            # 10. 装载到 1D 解向量
            uh1d_vec = bm.zeros(ndof_1d, dtype=bm.float64)
            # 利用编号模式: cell i 对应全局段 i*p .. i*p+p
            for i in range(E):
                seg = flat_vals[i*self.p : i*self.p + self.p + 1]
                uh1d_vec = bm.set_at(uh1d_vec, cell2dof_1d[i], seg)
            uh1d = fe_space.function(uh1d_vec)
            return int_mesh, uh1d, nodes_1d

        # 固定时间 t: 沿 x 方向
        if t is not None:
            built = _build_slice(t, fixed_axis=1, label='t')
            if built is not None:
                int_mesh, uh1d, _ = built
                exact = lambda x: self.pde.sl_solution(x, t)
                grad  = lambda x: self.pde.sl_gradient(x, t)
                L2e = int_mesh.error(exact, uh1d, q=self.q)
                H1e = int_mesh.error(grad,  uh1d.grad_value, q=self.q)
                self.logger.info(f"L2 Error: {L2e}, H1 Error: {H1e} at t={t}.")
                results['t'] = {'value': t, 'L2': float(L2e), 'H1': float(H1e)}

        # 固定空间 x=p: 沿 t 方向
        if p is not None:
            built = _build_slice(p, fixed_axis=0, label='x')
            if built is not None:
                int_mesh, uh1d, _ = built
                exact = lambda tau: self.pde.sl_solution(p, tau)
                grad  = lambda tau: self.pde.sl_gradient(p, tau)
                L2e = int_mesh.error(exact, uh1d, q=self.q)
                H1e = int_mesh.error(grad,  uh1d.grad_value, q=self.q)
                self.logger.info(f"L2 Error: {L2e}, H1 Error: {H1e} at x={p}.")
                results['x'] = {'value': p, 'L2': float(L2e), 'H1': float(H1e)}

        if return_error:
            if not results:
                return None
            return results
        
    def interpolate_error(self):
        solution = self.pde.solution
        gradient = self.pde.gradient
        space = self.space
        sol_int = space.interpolate(solution)

        mesh = self.mesh
        L2e = mesh.error(solution, sol_int, q=self.q)
        H1e = mesh.error(gradient, sol_int.grad_value, q=self.q)
        self.logger.info(f"L2 Error: {L2e}, H1 Error: {H1e} at interpolation.")

    def to_vtk(self, filename='parabolic_stfem_solution.vtu'):
        uh = self.uh
        mesh = self.mesh
        mesh.nodedata['uh_node'] = uh
        mesh.celldata['uh'] = uh.value(self.bcs)
        mesh.to_vtk(filename)

        