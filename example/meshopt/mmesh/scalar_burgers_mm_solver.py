from fealpy.backend import bm
from fealpy.fem import (ScalarDiffusionIntegrator,ScalarConvectionIntegrator,
                        ScalarSourceIntegrator,ScalarMassIntegrator)
from fealpy.fem import LinearForm, BilinearForm , DirichletBC
from fealpy.functionspace import LagrangeFESpace
from fealpy.mmesh import MMesher, Config
from fealpy.decorator import barycentric
from fealpy.solver import spsolve
from fealpy.mmesh.pde.scalar_burgers_data import ScalarBurgersData
import matplotlib.pyplot as plt
import argparse
import numpy as np

class Burgers_MMsolver:
    def __init__(self, pde: ScalarBurgersData ,p = 1 , 
                 nt = 500 , method = 'default', sub_steps=4,
                 mm_gamma=1.0, mm_tau=0.004, mm_t_max=0.1,
                 monitor='linear_int_error', mm_steps=10,
                 mm_tol=None, plot=True, log_interp_error=False,
                 log_mesh_displacement=False, bdf_max_steps=40,
                 bdf_max_stage_failures=8):
        """
        标量Burgers方程移动网格求解器, 时间积分采用SDIRK2方法
        u_t + u u_x + u u_y - 1/Re (u_xx + u_yy) = f
        
        Parameters
            pde : ScalarBurgersData
               标量Burgers方程数据对象
            p : int
                有限元空间的多项式次数
            nt : int
                时间步数
            method : str
                移动网格方法
        """
        self.pde = pde
        self.nt = nt
        self.method = method
        self.mm_gamma = mm_gamma
        self.mm_tau = mm_tau
        self.mm_t_max = mm_t_max
        self.monitor = monitor
        self.mm_steps = mm_steps
        self.mm_tol = mm_tol
        self.plot = plot
        self.log_interp_error = log_interp_error
        self.log_mesh_displacement = log_mesh_displacement
        self.bdf_max_steps = bdf_max_steps
        self.bdf_max_stage_failures = bdf_max_stage_failures
        
        self.dt = (pde.T[1] - pde.T[0]) / nt
        self.p = p
        self.q = p + 2
        self.mesh = pde.mesh
        self.Re = pde.Re
        
        gamma = 1- bm.sqrt(2)/2 
        self.tau1 = gamma
        self.tau2 = 1
        self.a11 = gamma
        self.a21 = 1- gamma
        self.a22 = gamma
        self.b1 = 1 - gamma
        self.b2 = gamma
        self.sub_steps = sub_steps
    
    def linear_system(self):
        """
        线性系统的组装和函数空间的定义
        """
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.bform0 = BilinearForm(self.space)
        self.bform1 = BilinearForm(self.space)
        self.lform = LinearForm(self.space)
        
        # 积分子定义
        self.SSI = ScalarSourceIntegrator(q=self.q)
        self.SMI = ScalarMassIntegrator(q=self.q)
        self.SDI = ScalarDiffusionIntegrator(q=self.q)
        self.SCI = ScalarConvectionIntegrator(q=self.q)
        
        self.bform0.add_integrator(self.SDI,self.SCI)
        self.bform1.add_integrator(self.SMI)
        self.lform.add_integrator(self.SSI)
        
        self.uh = self.space.function()
        self.u1 = self.space.function()
        self.u2 = self.space.function()
        
        self.bc = DirichletBC(self.space)
    
    def moving_mesher(self):
        """
        移动网格方法的初始化
        """
        method = self.method
        
            
        mesh = self.mesh
        space = self.space
        config = Config()
        if method == 'default':
            config.active_method = 'GFMMPDE'
        else:
            config.active_method = method
        config.is_pre = False
        # config.pde = pde
        config.mol_times = 6
        config.pre_steps = 4
        config.alpha = 0.5
        config.tau = self.mm_tau
        config.t_max = self.mm_t_max
        config.tol = self.mm_tol
        config.gamma = self.mm_gamma
        uh0 = space.interpolate(self.pde.init_solution)
        self.mm = MMesher(mesh,uh=uh0, space=space,beta=0.5, config=config)
        self.mm.initialize()
        self.mm.set_interpolation_method('linear')
        self.mm.set_monitor(self.monitor)
        self.mm.set_mol_method('huangs_method')
        self.mm.instance.total_steps = self.mm_steps
        self.mm.instance.bdf_max_steps = self.bdf_max_steps
        self.mm.instance.bdf_max_stage_failures = self.bdf_max_stage_failures
        if method in {'MetricTensorAdaptive', 'EAGAdaptiveHuang'}:
            self.mm.process = lambda: self.mm.instance.mesh_redistributor(
                method='BDF_SMW'
            )
        self.mspace = self.mm.instance.mspace
    
    def update(self,uh , t ,mv , sub_steps=None):
        if sub_steps is None:
            sub_steps = self.sub_steps
        delta = self.dt / sub_steps
        mesh = self.mesh
        a = 1 / self.Re
        SDI = self.SDI
        SCI = self.SCI
        SSI = self.SSI
        SMI = self.SMI
        bc = self.bc
        space = self.space
        node0 = mesh.node - mv * self.dt
        v0 = self.mspace.function(mv[:,0])
        v1 = self.mspace.function(mv[:,1])
        
        for j in range(sub_steps):
            t_hat = t + j * delta
            mesh.node = node0 + self.tau1 * mv * delta
            M = self.bform1.assembly()
            
            SDI.coef = a * delta * self.a11
            SMI.coef = 1.0
            
            @barycentric
            def coef1(bcs, index):
                v0_val = v0(bcs, index)
                v1_val = v1(bcs, index)
                v_value = bm.concat([v0_val[...,None], v1_val[...,None]], axis=-1)
                return -delta * self.a11 * v_value
            SCI.coef = coef1
            @barycentric
            def source1(bcs , index):
                guh = uh.grad_value(bcs , index)
                result = -guh[...,0]  - guh[...,1]
                result *= delta * self.a11 * uh(bcs , index)
                result += uh(bcs , index)
                return result
            SSI.source = source1
            
            A = self.bform0.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.tau1 * delta)
            A,b = bc.apply(A , b)
            self.u1[:] = spsolve(A , b , 'scipy')
            
            mesh.node = node0 + delta * mv
            k1 = (self.u1 - uh) / (self.tau1 * delta)
            k1 = space.function(k1)

            SDI.coef = a * delta * self.a22
            SMI.coef = 1.0
            
            @barycentric
            def coef2(bcs, index):
                v0_val = v0(bcs, index)
                v1_val = v1(bcs, index)
                v_value = bm.concat([v0_val[...,None], v1_val[...,None]], axis=-1)
                return - delta * self.a22 * v_value
            SCI.coef = coef2
            @barycentric
            def source2(bcs , index):
                guh = uh.grad_value(bcs , index)
                result = -guh[...,0] - guh[...,1]
                result *= delta * self.a22 * uh(bcs , index)
                result += uh(bcs , index) + delta * self.a21 * k1(bcs , index)
                return result
            SSI.source = source2
            A = self.bform0.assembly()
            M = self.bform1.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.tau2 * delta)
            A,b = bc.apply(A , b)
            self.u2[:] = spsolve(A , b , 'scipy')
            
            k2 = (self.u2 - uh - delta * self.a21 * k1) / (self.a22 * delta)
            self.uh[:] = uh + delta * (self.b1 * k1 + self.b2 * k2)
            
            node0 = mesh.node.copy()
            uh[:] = self.uh[:]
        
        return uh
    
    def error(self,uh , t):
        mesh = self.mesh
        pde = self.pde
        L2_error = mesh.error(uh,lambda p : pde.solution(p,t+self.dt),power = 2)
        uh_grad = uh.grad_value
        
        H1_error = mesh.error(uh_grad,lambda p : pde.gradient(p,t+self.dt),power = 2)
        
        return L2_error , H1_error

    def exact_interpolation_error(self, t):
        uh = self.space.interpolate(lambda p: self.pde.solution(p, t))
        return self.mesh.error(uh, lambda p: self.pde.solution(p, t), power=2)

    def displacement_stats(self, old_node, new_node):
        d = np.linalg.norm(np.asarray(new_node) - np.asarray(old_node), axis=1)
        nx = getattr(self.pde, "nx", None)
        ny = getattr(self.pde, "ny", None)
        if nx is None or ny is None:
            area = (self.pde.D[1] - self.pde.D[0]) * (self.pde.D[3] - self.pde.D[2])
            h = np.sqrt(area / max(1, self.mesh.number_of_cells()))
        else:
            h = min((self.pde.D[1] - self.pde.D[0]) / nx,
                    (self.pde.D[3] - self.pde.D[2]) / ny)
        return {
            "max": d.max(),
            "mean": d.mean(),
            "rms": np.sqrt(np.mean(d*d)),
            "q95": np.quantile(d, 0.95),
            "q99": np.quantile(d, 0.99),
            "h": h,
        }

    def format_displacement_stats(self, stats):
        h = stats["h"]
        return (
            f"max={stats['max']:.6e} ({stats['max']/h:.3f}h), "
            f"mean={stats['mean']:.6e} ({stats['mean']/h:.3f}h), "
            f"rms={stats['rms']:.6e} ({stats['rms']/h:.3f}h), "
            f"q95={stats['q95']:.6e}, q99={stats['q99']:.6e}"
        )

    def run_mesher(self, t):
        old_node = self.mesh.node.copy()
        old_error = self.exact_interpolation_error(t)
        if self.method != 'EAGAdaptiveHuang':
            self.mm.run()
            if self.log_interp_error:
                new_error = self.exact_interpolation_error(t)
                print(
                    f"Interpolation L2 at t={t:.6f}: "
                    f"{old_error:.6e} -> {new_error:.6e}"
                )
            self.log_displacement(t, old_node, self.mesh.node)
            return

        old_uh = self.uh.copy()
        old_mm_uh = self.mm.instance.uh.copy()

        try:
            self.mm.run()
        except Exception as exc:
            print(f"Rejecting EAGAdaptiveHuang mesh move: {exc}", flush=True)
            self.mm.instance._construct(old_node)
            self.uh[:] = old_uh
            self.mm.instance.uh[:] = old_mm_uh
            self.log_displacement(t, old_node, self.mesh.node, rejected=True)
            return

        new_error = self.exact_interpolation_error(t)
        if self.log_interp_error:
            print(
                f"Interpolation L2 at t={t:.6f}: "
                f"{old_error:.6e} -> {new_error:.6e}"
            )
        if new_error <= max(1.25 * old_error, old_error + 1e-10):
            self.log_displacement(t, old_node, self.mesh.node)
            return

        print(
            "Rejecting EAGAdaptiveHuang mesh move: "
            f"exact interpolation L2 {new_error:.6e} > {old_error:.6e}",
            flush=True,
        )
        self.mm.instance._construct(old_node)
        self.uh[:] = old_uh
        self.mm.instance.uh[:] = old_mm_uh
        self.log_displacement(t, old_node, self.mesh.node, rejected=True)

    def log_displacement(self, t, old_node, new_node, rejected=False):
        if not self.log_mesh_displacement:
            return
        status = "rejected" if rejected else "accepted"
        print(
            f"Mesh displacement at t={t:.6f} ({status}): "
            f"{self.format_displacement_stats(self.displacement_stats(old_node, new_node))}"
        )

    def run_initial_mesh_test(self, t=0.0, total_time=1.0, chunk_time=5.0e-5,
                              max_chunks=200, reject_factor=1.25,
                              error_cap=None, shrink=0.5, min_chunk=1.0e-8,
                              stop_at_target=None, log_every=1):
        self.linear_system()
        self.moving_mesher()
        self.uh = self.space.interpolate(self.pde.init_solution)
        self.mm.update_solution(self.uh)

        inst = self.mm.instance
        initial_node = self.mesh.node.copy()
        total_done = 0.0
        chunk = float(chunk_time)
        accepted = 0
        rejected = 0
        step = 0

        while total_done < total_time and step < max_chunks:
            step += 1
            chunk = min(chunk, total_time - total_done)
            old_node = self.mesh.node.copy()
            old_uh = self.uh.copy()
            old_mm_uh = inst.uh.copy()
            old_t_span = inst.t_span
            old_t_max = inst.t_max
            old_config_t_max = inst.config.t_max
            old_error = self.exact_interpolation_error(t)

            inst.t_span = chunk
            inst.t_max = chunk
            inst.config.t_max = chunk
            failed = None
            try:
                inst.mesh_redistributor(total_steps=1, method='BDF_SMW')
            except Exception as exc:
                failed = exc
                inst._construct(old_node)
                self.uh[:] = old_uh
                inst.uh[:] = old_mm_uh
            finally:
                inst.t_span = old_t_span
                inst.t_max = old_t_max
                inst.config.t_max = old_config_t_max

            new_error = self.exact_interpolation_error(t)
            stats = self.displacement_stats(old_node, self.mesh.node)
            cap_ok = error_cap is not None and new_error <= error_cap
            factor_ok = new_error <= max(reject_factor * old_error, old_error + 1e-10)
            ok = failed is None and (cap_ok or factor_ok)

            if ok:
                accepted += 1
                total_done += chunk
                self.uh[:] = inst.uh[:]
                status = "accepted"
            else:
                rejected += 1
                inst._construct(old_node)
                self.uh[:] = old_uh
                inst.uh[:] = old_mm_uh
                status = "failed" if failed is not None else "rejected"
                chunk = max(min_chunk, shrink * chunk)

            should_log = log_every <= 1 or step == 1 or step % log_every == 0 or not ok
            if should_log:
                print(
                    f"Initial mesh chunk {step}: {status}, "
                    f"done={total_done:.6e}/{total_time:.6e}, chunk={chunk:.6e}, "
                    f"interp={old_error:.6e}->{new_error:.6e}, "
                    f"{self.format_displacement_stats(stats)}",
                    flush=True,
                )
                if failed is not None:
                    print(f"  failure: {failed}", flush=True)

            if stop_at_target is not None and ok and new_error >= stop_at_target:
                print(
                    f"Initial mesh target reached: interp={new_error:.6e} "
                    f">= {stop_at_target:.6e}"
                )
                break
            if not ok and chunk <= min_chunk:
                print(f"Initial mesh chunk fell to min_chunk={min_chunk:.6e}; stopping.")
                break

        final_error = self.exact_interpolation_error(t)
        total_stats = self.displacement_stats(initial_node, self.mesh.node)
        print(
            "Initial mesh summary: "
            f"accepted={accepted}, rejected={rejected}, "
            f"done={total_done:.6e}/{total_time:.6e}, "
            f"interp={final_error:.6e}, "
            f"{self.format_displacement_stats(total_stats)}"
        )
        return {
            "accepted": accepted,
            "rejected": rejected,
            "total_time": total_done,
            "interp_error": final_error,
            "displacement": total_stats,
        }
    
    def _plot_errors(self, L2_list, H1_list, times):
        """绘制 L2 和 H1 误差随时间的变化曲线"""
        if not self.plot:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # L2 误差曲线
        ax1.semilogy(times, L2_list, 'b-o', linewidth=2, markersize=4, label='L2 Error')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('L2 Error', fontsize=12)
        ax1.set_title('L2 Error vs Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # H1 误差曲线
        ax2.semilogy(times, H1_list, 'r-s', linewidth=2, markersize=4, label='H1 Error')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('H1 Error', fontsize=12)
        ax2.set_title('H1 Error vs Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'burgers_errors_{self.method}_nt{self.nt}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def solve(self, vtu_path=None):
        self.linear_system()
        self.moving_mesher()
        pde = self.pde
        nt = self.nt
        dt = self.dt
        space = self.space
        mesh = self.mesh
        mm = self.mm
        times = bm.linspace(pde.T[0], pde.T[1], nt + 1)
        self.run_mesher(times[0])

        self.uh = space.interpolate(pde.init_solution)
        sub_steps = self.sub_steps * 3
        L2_list = []
        H1_list = []
        for i in range(nt):
            x0 = mesh.node.copy()
            if i>0:
                self.run_mesher(times[i])
                sub_steps = None
                
            mv = (mesh.node - x0)/dt
            t = times[i]
            self.uh[:] = self.update(self.uh , t , mv , sub_steps=sub_steps)
            L2_error , H1_error = self.error(self.uh , t)
            L2_list.append(L2_error)
            H1_list.append(H1_error)
            mm.update_solution(self.uh)
            if vtu_path is not None:
                mesh.nodedata['u'] = self.uh[:]
                mesh.to_vtk(f'{vtu_path}_step{i+1:04d}.vtu')
            print(f'Step {i+1}/{nt}, Time={t+dt:.4f}, L2 Error={L2_error:.6e}, H1 Error={H1_error:.6e}')
        
        self._plot_errors(L2_list, H1_list, times[1:])
        return self.uh, L2_list, H1_list    

def monitor_name(name):
    aliases = {
        'hessian': 'linear_int_error',
        'linear_int_error': 'linear_int_error',
        'arclength': 'matrix_arc_length',
        'arc_length': 'matrix_arc_length',
        'matrix_arc_length': 'matrix_arc_length',
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown monitor/metric '{name}'. "
            "Use hessian, linear_int_error, arclength, or matrix_arc_length."
        ) from exc

def method_name(functional):
    aliases = {
        'ours': 'MetricTensorAdaptive',
        'trace_log': 'MetricTensorAdaptive',
        'huang': 'EAGAdaptiveHuang',
        'MetricTensorAdaptive': 'MetricTensorAdaptive',
        'EAGAdaptiveHuang': 'EAGAdaptiveHuang',
    }
    try:
        return aliases[functional]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown functional/method '{functional}'. "
            "Use ours, huang, MetricTensorAdaptive, or EAGAdaptiveHuang."
        ) from exc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='MetricTensorAdaptive',
                        type=method_name)
    parser.add_argument('--functional', choices=('ours', 'huang'),
                        help='Paper functional alias. Overrides --method.')
    parser.add_argument('--monitor', '--metric', default='linear_int_error',
                        type=monitor_name)
    parser.add_argument('--nt', default=1000, type=int)
    parser.add_argument('--nx', default=30, type=int)
    parser.add_argument('--ny', default=30, type=int)
    parser.add_argument('--sub-steps', default=4, type=int)
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--tau', default=None, type=float)
    parser.add_argument('--t-max', default=0.1, type=float)
    parser.add_argument('--start-time', default=0.0, type=float)
    parser.add_argument('--final-time', default=2.0, type=float)
    parser.add_argument('--mm-steps', default=10, type=int)
    parser.add_argument('--mm-tol', default=None, type=float)
    parser.add_argument('--bdf-max-steps', default=40, type=int)
    parser.add_argument('--bdf-max-stage-failures', default=8, type=int)
    parser.add_argument('--vtu-path', default=None)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--log-interp-error', action='store_true')
    parser.add_argument('--log-mesh-displacement', action='store_true')
    parser.add_argument('--initial-mesh-only', action='store_true')
    parser.add_argument('--artificial-total-time', default=1.0, type=float)
    parser.add_argument('--artificial-chunk-time', default=None, type=float)
    parser.add_argument('--initial-max-chunks', default=200, type=int)
    parser.add_argument('--interp-reject-factor', default=1.25, type=float)
    parser.add_argument('--interp-error-cap', default=None, type=float)
    parser.add_argument('--initial-stop-at-target', default=None, type=float)
    parser.add_argument('--chunk-shrink', default=0.5, type=float)
    parser.add_argument('--min-chunk-time', default=1.0e-8, type=float)
    parser.add_argument('--initial-log-every', default=1, type=int)
    args = parser.parse_args()
    if args.functional is not None:
        args.method = method_name(args.functional)
    if args.gamma is None:
        args.gamma = 1.5 if args.method == 'EAGAdaptiveHuang' else 1.25
    if args.tau is None:
        args.tau = 0.01 if args.method == 'EAGAdaptiveHuang' else 0.004

    if args.final_time <= args.start_time:
        raise ValueError("--final-time must be larger than --start-time")

    pde = ScalarBurgersData(u, var, D, [args.start_time, args.final_time], Re=Re)
    pde.set_mesh(nx=args.nx, ny=args.ny, meshtype='cross_tri')
    solver = Burgers_MMsolver(pde, p=1, nt=args.nt,
                              method=args.method,
                              sub_steps=args.sub_steps,
                              mm_gamma=args.gamma,
                              mm_tau=args.tau,
                              mm_t_max=args.t_max,
                              monitor=args.monitor,
                              mm_steps=args.mm_steps,
                              mm_tol=args.mm_tol,
                              plot=not args.no_plot,
                              log_interp_error=args.log_interp_error,
                              log_mesh_displacement=args.log_mesh_displacement,
                              bdf_max_steps=args.bdf_max_steps,
                              bdf_max_stage_failures=args.bdf_max_stage_failures)
    if args.initial_mesh_only:
        chunk_time = args.artificial_chunk_time
        if chunk_time is None:
            chunk_time = args.t_max
        solver.run_initial_mesh_test(
            t=args.start_time,
            total_time=args.artificial_total_time,
            chunk_time=chunk_time,
            max_chunks=args.initial_max_chunks,
            reject_factor=args.interp_reject_factor,
            error_cap=args.interp_error_cap,
            shrink=args.chunk_shrink,
            min_chunk=args.min_chunk_time,
            stop_at_target=args.initial_stop_at_target,
            log_every=args.initial_log_every,
        )
        return
    solver.solve(vtu_path=args.vtu_path)


Re = 100
u = f'1/(1+ exp(({Re}/2)*(x+y-t)))'
var = ['x', 'y', 't']
D = [0, 1, 0, 1]
T = [0, 2]
support_method = ['default', 'GFMMPDE', 'Harmap',
                  'EAGAdaptiveHuang', 'EAGAdaptiveXHuang',
                  'MetricTensorAdaptive', 'MetricTensorAdaptiveX']

if __name__ == '__main__':
    main()
