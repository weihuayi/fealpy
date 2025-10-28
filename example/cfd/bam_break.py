from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
bm.set_backend("numpy")
from fealpy.mesh.node_mesh import BamBreak
from fealpy.cfd.simulation.sph.particle_solver_new import ParticleSystem, BamBreakSolver, Visualizer

dx = 0.03
dy = 0.03
maxstep = 2000

# 创建网格
mesh = BamBreak.from_bam_break_domain(dx, dy)
# 创建粒子系统
particle_system = ParticleSystem.initialize_particles(mesh.node, mesh.nodedata)

# 创建SPH求解器
sph_solver = BamBreakSolver(particle_system)

# 初始可视化
color = bm.where(particle_system.particles["tag"], "red", "blue")
plt.scatter(particle_system.particles["position"][:, 0], 
            particle_system.particles["position"][:, 1], 
            c=color, s=5)
plt.grid(True)
plt.show()

# 运行模拟
sph_solver.run_simulation(maxstep, draw_interval=10, reinitalize_interval=30)

# 最终结果可视化
visualizer = Visualizer(particle_system.particles["position"]
                        , particle_system.particles["pressure"]
                        , particle_system.particles["tag"])
    
visualizer.final_plot()



