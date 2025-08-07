from ..backend import backend_manager as bm
from ..opt import *
from ..opt.optimizer_base import opt_alg_options
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import UnivariateSpline

class UAVPathPlanning():
    """
    UAVPathPlanning encapsulates the entire UAV path planning process in a 3D environment with terrain data and threat avoidance.

    This class handles terrain modeling, threat modeling, path optimization using a customizable algorithm,
    and visualization of the resulting path and environment.

    Parameters:
        threats (array-like): An array of threats, each specified as [x, y, z, radius].
        terrain_data (array-like): A 2D terrain height map.
        start_pos (array-like): The UAV's starting 3D position.
        end_pos (array-like): The UAV's goal 3D position.
        opt_method (callable): A callable that returns an optimizer instance when passed an option dictionary.

    Attributes:
        MAPSIZE_X (int): Width of the terrain map.
        MAPSIZE_Y (int): Height of the terrain map.
        X, Y (Tensor): Meshgrid coordinates for terrain surface.
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D plot axis, initialized during plotting.
    """
    def __init__(self, threats, terrain_data, start_pos, end_pos, opt_method):
        self.threats = threats
        self.terrain_data = terrain_data
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.opt_method = opt_method
        self.MAPSIZE_X = terrain_data.shape[1]
        self.MAPSIZE_Y = terrain_data.shape[0]
        self.X, self.Y = bm.meshgrid(bm.arange(self.MAPSIZE_X), bm.arange(self.MAPSIZE_Y))
    
    def plot_mode(self):
        """
        Plots the terrain surface along with all defined threats (as vertical cylinders).

        Initializes a 3D matplotlib axis (self.ax) for later use.
        """
        fig = plt.figure(figsize=(12, 8))
        self.ax = fig.add_subplot(111, projection='3d')
        surf = self.ax.plot_surface(self.X, self.Y, self.terrain_data,
                             cmap=cm.summer,
                             linewidth=0,
                             antialiased=True,
                             shade=True,
                             alpha=0.5,
                             zorder=1)
        
        self.ax.set_position([0.05, 0.05, 0.9, 0.9])  # Adjust to prevent clipping
        self.ax.set_box_aspect([1, 1, 0.3])  # Z-axis scaled differently for better visualization
        self.ax.grid(False)

        h = 250

        for threat in self.threats:
            x, y, z, r = threat
            theta = bm.linspace(0, 2*bm.pi, 30)
            z_cyl = bm.linspace(0, h, 10)
            theta_grid, z_grid = bm.meshgrid(theta, z_cyl)

            x_cyl = r * bm.cos(theta_grid) + x
            y_cyl = r * bm.sin(theta_grid) + y
            z_cyl = z_grid + z

            self.ax.plot_surface(x_cyl, y_cyl, z_cyl,
                          color=[0.0078, 0.1882, 0.2784],  # RGB equivalent
                          alpha=0.7,
                          edgecolor='none',
                          zorder=10)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('UAV Path Planning Environment')
        self.ax.view_init(elev=30, azim=-45)
        self.ax.set_facecolor('white')
        plt.tight_layout()

    def output_solution(self, sol, smooth=1):
        """
        Visualizes the optimized UAV path over the 3D environment.

        Parameters:
            sol (Tensor): Optimized solution from the optimizer, of shape (1, 3*n).
            smooth (float): Spline smoothing parameter in [0, 1]. Default is 1 (interpolation).
        """
        self.plot_mode()
        x, y, z, _ = self.spherical_to_cart(sol)
        x_all = bm.concatenate(
            [
                bm.array([[self.start_pos[0]]]),
                x,
                bm.array([[self.end_pos[0]]])
            ], axis=1
        )
        y_all = bm.concatenate(
            [
                bm.array([[self.start_pos[1]]]),
                y,
                bm.array([[self.end_pos[1]]])
            ], axis=1
        )
        z_all = bm.concatenate(
            [
                bm.array([[self.start_pos[2]]]),
                z,
                bm.array([[self.end_pos[2]]])
            ], axis=1
        )

        route = bm.stack([x_all, y_all, z_all], axis=-1).squeeze()
        diff = route[:-1] - route[1:]
        total_distance = bm.sum(bm.linalg.norm(diff, axis=1))

        print('UAV Route: ')
        print(route)
        print('Total Route Distance: ', total_distance)

        H = self.terrain_data
        y_index = bm.round(y_all).astype(int)
        x_index = bm.round(x_all).astype(int)
        z_all = H[y_index, x_index] + z_all
        xyz = bm.concatenate([x_all, y_all, z_all])
        npts = xyz.shape[1]
        t = bm.arange(npts)
        s = (1 - smooth) * npts
        xyzp = bm.array([
            UnivariateSpline(t, xyz_k, s=s)(t)
            for xyz_k in xyz
        ], dtype=float)

        self.ax.plot(
            xyzp[0, :],
            xyzp[1, :],
            xyzp[2, :],
            color='k',       
            linewidth=1,     
            marker=None,      
        )
        self.ax.scatter([x_all[0][0]], [y_all[0][0]], [z_all[0][0]], color='blue', label='Start')
        self.ax.scatter([x_all[-1][-1]], [y_all[-1][-1]], [z_all[-1][-1]], color='red', label='End')
        self.ax.legend(loc='upper right') 
        plt.show()


    def opt(self, n):
        """
        Performs the UAV trajectory optimization process.

        Parameters:
            n (int): Number of trajectory segments.

        Returns:
            Tuple: (optimal_path, optimal_cost)
                - optimal_path (Tensor): Best solution found, shape (1, 3*n).
                - optimal_cost (float): Corresponding cost.
        """
        self.n = n
        self.prepare_uav_bounds()
        NP = 50
        dim = 3 * n
        fun = lambda x: self.cost_function(x)
        x0 = initialize(NP, dim, self.ub, self.lb, method=None)
        option = opt_alg_options(x0, fun, (self.lb, self.ub), NP, MaxIters=100)
        self.optimizer = self.opt_method(option)
        self.optimizer.run()
        
        return self.optimizer.gbest[None, :], self.optimizer.gbest_f

    def spherical_to_cart(self, sol):
        """
        Converts spherical parameters into 3D Cartesian coordinates, applying penalties for constraint violations.

        Parameters:
            sol (Tensor): Input tensor of shape (batch_size, 3*n).

        Returns:
            Tuple: (x, y, z, F4)
                - x, y, z (Tensor): Cartesian coordinates of shape (batch_size, n).
                - F4 (Tensor): Penalty values due to unsafe pitch angles, shape (batch_size,).
        """
        F_inf = 1e+10  # Large penalty value
        F4 = bm.zeros((sol.shape[0]))  # Penalty accumulator

        n = int(sol.shape[1] / 3)  # Number of trajectory segments

        # Allocate space for output coordinates
        x = bm.zeros((sol.shape[0], n))
        y = bm.zeros((sol.shape[0], n))
        z = bm.zeros((sol.shape[0], n))

        # Split the input into r, delta_psi, and delta_phi components
        r = sol[:, :n]
        delta_psi = sol[:, n:2*n]
        delta_phi = sol[:, 2*n:3*n]

        # Initialize the first point based on the star position
        x[:, 0] = self.start_pos[0] + r[:, 0] * bm.cos(delta_psi[:, 0]) * bm.sin(delta_phi[:, 0])
        y[:, 0] = self.start_pos[1] + r[:, 0] * bm.cos(delta_psi[:, 0]) * bm.cos(delta_phi[:, 0])
        z[:, 0] = self.start_pos[2] + r[:, 0] * bm.sin(delta_psi[:, 0])

        # Initial direction vector from the start to the first point
        dir_vector = bm.concatenate([
            x[:, 0][:, None], y[:, 0][:, None], z[:, 0][:, None]
        ], axis=1) - self.start_pos  # shape: (batch_size, 3)

        # Compute initial azimuth (phi0) and elevation (psi0) angles
        phi0 = bm.pi / 2 - bm.atan2(dir_vector[:, 1], dir_vector[:, 0])  # azimuth
        horizontal_dist = bm.linalg.norm(dir_vector[:, :2], axis=1)      # distance in xy-plane
        psi0 = bm.atan2(dir_vector[:, 2], horizontal_dist)               # elevation

        # Apply penalty if elevation exceeds ±π/7
        mask = (psi0 > bm.pi / 4) + (psi0 < -bm.pi / 4)
        F4[mask] = F4[mask] + F_inf

        # Iterate over the remaining segments
        for i in range(1, n):
            phi = phi0 + delta_phi[:, i]
            psi = psi0 + delta_psi[:, i]

            # Update coordinates using spherical to Cartesian conversion
            x[:, i] = x[:, i-1] + r[:, i] * bm.cos(psi) * bm.sin(phi)
            y[:, i] = y[:, i-1] + r[:, i] * bm.cos(psi) * bm.cos(phi)
            z[:, i] = z[:, i-1] + r[:, i] * bm.sin(psi)

            # Compute direction vector from previous to current point
            dir_vector = bm.concatenate([
                x[:, i][:, None], y[:, i][:, None], z[:, i][:, None]
            ], axis=1) - bm.concatenate([
                x[:, i-1][:, None], y[:, i-1][:, None], z[:, i-1][:, None]
            ], axis=1)

            # Update azimuth and elevation angles
            phi0 = bm.pi / 2 - bm.atan2(dir_vector[:, 1], dir_vector[:, 0])
            horizontal_dist = bm.linalg.norm(dir_vector[:, :2], axis=1)
            psi0 = bm.atan2(dir_vector[:, 2], horizontal_dist)

            # Apply penalty again if elevation constraint is violated
            mask = (psi0 > bm.pi / 4) + (psi0 < -bm.pi / 4)
            F4[mask] = F4[mask] + F_inf

        x = bm.clip(x, 0, self.terrain_data.shape[1]-1)
        y = bm.clip(y, 0, self.terrain_data.shape[0]-1)

        return x, y, z, F4

    def dist_point_to_segment(self, P, A, B):
        """
        Computes the perpendicular distance from point(s) P to line segments AB.

        Parameters:
            P (Tensor): Points of shape (..., 2).
            A (Tensor): Segment start points of shape (..., 2).
            B (Tensor): Segment end points of shape (..., 2).

        Returns:
            Tensor: Distances from P to segments AB, shape (...,).
        """
        AP = P - A
        AB = B - A
        AB_norm_sq = bm.sum(AB**2, axis=-1, keepdims=True)

        # Avoid divide-by-zero
        t = bm.sum(AP * AB, axis=-1, keepdims=True) / (AB_norm_sq + 1e-8)
        t = bm.clip(t, 0.0, 1.0)

        proj = A + t * AB
        return bm.linalg.norm(P - proj, axis=-1)

    def cost_function(self, sol):
        """
        Computes the total cost of a trajectory using multiple penalty components.

        Cost is the weighted sum of:
            - F1: Path length
            - F2: Threat avoidance
            - F3: Altitude safety
            - F4: Angle safety (from spherical_to_cart)

        Parameters:
            sol (Tensor): Solution tensor of shape (batch_size, 3*n).

        Returns:
            Tensor: Total cost of each batch solution, shape (batch_size,).
        """
        x, y, z, F4 = self.spherical_to_cart(sol)
        F_inf = 1e+10
        z_min = 100
        z_max = 200
        x_all = bm.concatenate(
            [
                bm.ones((sol.shape[0], 1))*self.start_pos[0],
                x,
                bm.ones((sol.shape[0], 1))*self.end_pos[0]
            ], axis=1
        )
        y_all = bm.concatenate(
            [
                bm.ones((sol.shape[0], 1))*self.start_pos[1],
                y,
                bm.ones((sol.shape[0], 1))*self.end_pos[1]
            ], axis=1
        )
        z_all = bm.concatenate(
            [
                bm.ones((sol.shape[0], 1))*self.start_pos[2],
                z,
                bm.ones((sol.shape[0], 1))*self.end_pos[2]
            ], axis=1
        )
        H = self.terrain_data
        y_index = bm.round(y_all).astype(int)
        x_index = bm.round(x_all).astype(int)
        z_abs = H[y_index, x_index] + z_all

        batch_size, N = x_all.shape

        dx = x_all[:, 1:] - x_all[:, :-1]
        dy = y_all[:, 1:] - y_all[:, :-1]
        dz = z_abs[:, 1:] - z_abs[:, :-1]
        diffs = bm.stack([dx, dy, dz], axis=2)
        norms = bm.linalg.norm(diffs, axis=2)
        F1 = bm.sum(norms, axis=1)

        drone_size = 10
        danger_dist = 2 * drone_size
        buffer = drone_size + danger_dist
        F2 = bm.zeros(batch_size)
        for i in range(self.threats.shape[0]):
            threat = self.threats[i]
            threat_pos = bm.array([threat[0], threat[1]])  
            threat_radius = threat[3]
            safe_dist = threat_radius + buffer
            collision_dist = threat_radius + drone_size

            P1 = bm.stack([x_all[:, :-1], y_all[:, :-1]], axis=2) 
            P2 = bm.stack([x_all[:, 1:], y_all[:, 1:]], axis=2)   

            P_threat = bm.broadcast_to(threat_pos[None, None, :], (batch_size, N - 1, 2))
            dists = self.dist_point_to_segment(P_threat, P1, P2)  

            cost = bm.where(
                dists > safe_dist, 
                0.0,
                bm.where(dists < collision_dist, F_inf, safe_dist - dists)
            )

            F2 += bm.sum(cost, axis=1) 
        
        mask = (z < 0) | (z < z_min) | (z > z_max) 
        z_mid = (z_max + z_min) / 2
        f3 = bm.where(mask, F_inf, bm.abs(z - z_mid))
        F3 = bm.sum(f3, axis=1)

        b1, b2, b3, b4 = 10, 100, 10, 10
        cost = b1 * F1 + b2 * F2 + b3 * F3 + b4 * F4
        return cost

    def prepare_uav_bounds(self, angle_range=bm.pi/4):
        """
        Prepares upper and lower bounds for spherical parameters based on UAV dynamics and angle constraints.

        Parameters:
            angle_range (float): Maximum allowed deviation from initial direction, default is pi/4.
        """
        r_max = bm.linalg.norm(self.start_pos - self.end_pos) 
        r_min = 0
        psi_max = angle_range
        psi_min = -angle_range

        phi_max = angle_range
        phi_min = -angle_range

        dir_vector = self.end_pos - self.start_pos
        phi0 = bm.pi / 2 - bm.atan2(dir_vector[1], dir_vector[0])
        self.ub = bm.concatenate([
            bm.ones(self.n) * r_max,
            bm.ones(self.n) * psi_max,
            bm.array([phi0 + bm.pi/2]),
            bm.ones(self.n - 1) * phi_max
        ])
        self.lb = bm.concatenate([
            bm.ones(self.n) * r_min,
            bm.ones(self.n) * psi_min,
            bm.array([phi0 - bm.pi/2]),
            bm.ones(self.n - 1) * phi_min
        ])
        
