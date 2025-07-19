from fealpy.backend import backend_manager as bm
import random
import matplotlib.pyplot as plt

class Point:
    """Represents a point on the map (either a target or the base)"""
    def __init__(self, x, y, weight=0, idx=-1):
        """
        Initialize a point

        Parameters:
            x (float): x-coordinate
            y (float): y-coordinate
            weight (int, optional): Point weight (importance of target). Defaults to 0.
            idx (int, optional): Point index (-1 indicates base). Defaults to -1.
        """
        self.x = x
        self.y = y
        self.weight = weight
        self.idx = idx

class WeightTargetsSweepCoverageAlg:
    """UAV path planning system for multi-drone cooperative patrol path planning"""
    
    def __init__(self):
        """Initialize the path planner"""
        self.targets = []   # List of target points
        self.base = None    # Base location
        self.paths = []     # Planned paths
        self.coverage = 0.0 # Target coverage ratio
        self.task_time = 0.0 # Total mission time
        
    def add_target(self, x, y, weight=1):
        """
        Add a target point

        Parameters:
            x (float): x-coordinate
            y (float): y-coordinate
            weight (int, optional): Target weight. Defaults to 1.
        """
        idx = len(self.targets)
        self.targets.append(Point(x, y, weight, idx))
        
    def set_base(self, x, y):
        """
        Set base location
        
        Parameters:
            x (float): Base x-coordinate
            y (float): Base y-coordinate
        """
        self.base = Point(x, y, 0, -1)
        
    def generate_random_scenario(self, num_targets, area_size, min_weight, max_weight):
        """
        Generate a random scenario
        
        Parameters:
            num_targets (int): Number of targets
            area_size (float): Area size (side length of square area)
            min_weight (int): Minimum target weight
            max_weight (int): Maximum target weight
        """
        # Set base at center
        self.set_base(area_size/2, area_size/2)
        
        # Generate random targets
        self.targets = []
        for i in range(num_targets):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            weight = random.randint(min_weight, max_weight)
            self.add_target(x, y, weight)
    
    @staticmethod
    def _distance(p1, p2):
        """
        Calculate Euclidean distance between two points
        
        Parameters:
            p1 (Point): First point
            p2 (Point): Second point
            
        Returns:
            float: Distance between points
        """
        return bm.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    @staticmethod
    def _angle_between(v1, v2):
        """
        Calculate angle between two vectors (in radians)
        
        Parameters:
            v1 (tuple): Vector 1 (x, y)
            v2 (tuple): Vector 2 (x, y)
            
        Returns:
            float: Angle between vectors (radians)
        """
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = bm.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = bm.sqrt(v2[0]**2 + v2[1]**2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_theta = dot / (norm1 * norm2)
        cos_theta = max(min(cos_theta, 1.0), -1.0)  # Avoid boundary issues
        return bm.acos(cos_theta)
    
    @staticmethod
    def _flight_time(p1, p2, prev_point, V, omega):
        """
        Calculate flight time from p1 to p2, considering turning angle
        
        Parameters:
            p1 (Point): Start point
            p2 (Point): Target point
            prev_point (Point): Previous point (for calculating turning angle)
            V (float): UAV speed (km/min)
            omega (float): UAV angular velocity (rad/min)
            
        Returns:
            float: Estimated flight time (minutes)
        """
        # Straight-line flight time
        linear_time = WeightTargetsSweepCoverageAlg._distance(p1, p2) / V
        
        # No turn needed if no previous point
        if prev_point is None:
            return linear_time
        
        # Calculate turning angle
        vec1 = (p1.x - prev_point.x, p1.y - prev_point.y)  # Previous segment vector
        vec2 = (p2.x - p1.x, p2.y - p1.y)                  # Current segment vector
        theta = WeightTargetsSweepCoverageAlg._angle_between(vec1, vec2)
        
        # Turning time
        turn_time = theta / omega
        
        return linear_time + turn_time
    
    def plan_paths(self, M, V, omega, T_fmax, T_s, O, alpha):
        """
        Execute the path planning algorithm
        
        Parameters:
            M (int): Number of UAVs
            V (float): UAV speed (km/h)
            omega (float): UAV angular velocity (rad/s)
            T_fmax (float): Maximum flight time (min)
            T_s (float): Single UAV setup time (min)
            O (int): Number of operators
            alpha (float): Balance factor between time and weight (0-1)
            
        Returns:
            dict: Contains planning results (paths, mission time, coverage, etc.)
        """
        # Unit conversions
        V = V / 60  # km/h → km/min
        omega = omega * 60  # rad/s → rad/min
        
        N = len(self.targets)
        # Initialize paths and flight times
        paths = [[] for _ in range(M)]  # Path for each UAV
        T_f = [0.0] * M                 # Flight time for each UAV
        T_c = [0.0] * M                 # Cumulative time for each UAV
        covered = [False] * N           # Coverage status for targets
        remaining_targets = set(range(N))  # Set of uncovered targets
        
        # Plan paths for each UAV sequentially
        for k in range(M):
            # Initialize current UAV's path
            path = [self.base]
            prev_point = None
            current_point = self.base
            
            # Calculate waiting time (operator preparation time)
            T_w = T_s * ((k + 1) / O)
            
            while remaining_targets and T_f[k] < T_fmax:
                # Calculate cost to each remaining target
                min_cost = float('inf')
                best_target = None
                best_time = 0.0
                
                for i in remaining_targets:
                    target = self.targets[i]
                    # Estimate cost using straight-line flight time
                    linear_time = WeightTargetsSweepCoverageAlg._distance(current_point, target) / V
                    cost = alpha * linear_time + (1 - alpha) / target.weight
                    
                    # Calculate full flight time (including turns)
                    ft = WeightTargetsSweepCoverageAlg._flight_time(current_point, target, prev_point, V, omega)
                    
                    # Check time constraint
                    if T_f[k] + ft <= T_fmax and cost < min_cost:
                        min_cost = cost
                        best_target = i
                        best_time = ft
                
                # If no valid target found, end current path
                if best_target is None:
                    break
                
                # Add target to path
                target = self.targets[best_target]
                path.append(target)
                
                # Update flight time and status
                T_f[k] += best_time
                covered[best_target] = True
                remaining_targets.remove(best_target)
                
                # Update point positions
                prev_point = current_point
                current_point = target
            
            # Safe backtracking: Try to return to base from points on path
            backtrack_success = False
            while len(path) > 1:  # At least one target in path
                # Try to return to base from current point
                return_time = WeightTargetsSweepCoverageAlg._flight_time(current_point, self.base, prev_point, V, omega)
                if T_f[k] + return_time <= T_fmax:
                    # Can safely return, add base and exit backtracking
                    path.append(self.base)
                    T_f[k] += return_time
                    backtrack_success = True
                    break
                else:
                    # Cannot return from current point, backtrack to previous
                    # Remove current point
                    removed_point = path.pop()
                    # Restore state: current point becomes previous
                    current_point = prev_point
                    # Update coverage status
                    if removed_point.idx >= 0:  # Ensure it's a target, not base
                        covered[removed_point.idx] = False
                        remaining_targets.add(removed_point.idx)
                    
                    # Update flight time: subtract flight time to removed point
                    if len(path) > 1:  # If path still has targets
                        # Calculate flight time to removed point
                        prev_prev_point = path[-2] if len(path) > 2 else self.base
                        remove_time = WeightTargetsSweepCoverageAlg._flight_time(
                            prev_prev_point, removed_point, 
                            path[-3] if len(path) > 3 else None, 
                            V, omega
                        )
                        T_f[k] -= remove_time
                        # Update prev_point
                        if len(path) > 2:
                            prev_point = path[-3] if len(path) > 3 else self.base
                        else:
                            prev_point = None
                    else:
                        # If backtracked to only base
                        T_f[k] = 0
                        prev_point = None
            
            # If still can't return after backtracking but path has targets
            if not backtrack_success and len(path) > 1:
                # Force return to base (may violate constraint)
                return_time = WeightTargetsSweepCoverageAlg._flight_time(current_point, self.base, prev_point, V, omega)
                path.append(self.base)
                T_f[k] += return_time
            
            # Save path and calculate cumulative time
            paths[k] = path
            T_c[k] = T_f[k] + T_w
        
        # Calculate total mission time and total weight
        T_task = max(T_c) if M > 0 else 0
        total_weight = sum(self.targets[i].weight for i in range(N) if covered[i])
        all_weight = sum(t.weight for t in self.targets)
        coverage = total_weight / all_weight if all_weight > 0 else 0.0
        
        # Save results
        self.paths = paths
        self.task_time = T_task
        self.coverage = coverage
        self.total_weight = total_weight
        
        return {
            "paths": paths,
            "task_time": T_task,
            "coverage": coverage,
            "covered_weight": total_weight,
            "all_weight": all_weight
        }
    
    def plot_paths(self, area_size=None):
        """
        Visualize paths and all targets
        
        Parameters:
            area_size (float, optional): Area size. Defaults to None.
        """
        if not self.paths or not self.base or not self.targets:
            print("无法绘制路径 - 请先设置基地和目标点")
            return
        
        plt.figure(figsize=(10, 10))
        colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c']  # Colors for different UAV paths
        
        # Get all covered targets
        covered_points = [p for path in self.paths for p in path if p.idx >= 0]
        covered_indices = {p.idx for p in covered_points}
        
        # Plot all targets
        all_x = [p.x for p in self.targets]
        all_y = [p.y for p in self.targets]
        all_weights = [p.weight for p in self.targets]
        all_sizes = [w * 10 for w in all_weights]  # Scale point size by weight
        
        # Different colors for covered vs uncovered targets
        covered_mask = [p.idx in covered_indices for p in self.targets]
        colors_all = ['green' if covered else 'gray' for covered in covered_mask]
        alphas_all = [0.9 if covered else 0.4 for covered in covered_mask]
        
        plt.scatter(
            all_x, all_y,
            s=all_sizes,
            c=colors_all,
            alpha=alphas_all,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Plot paths
        for k, path in enumerate(self.paths):
            if len(path) > 1:  # Has actual path
                color = colors[k % len(colors)]
                xs = [p.x for p in path]
                ys = [p.y for p in path]
                plt.plot(xs, ys, color=color, linewidth=1.5, marker='o', markersize=4, 
                         label=f'UAV {k+1}')
        
        # Plot base
        plt.scatter(self.base.x, self.base.y, s=200, c='red', marker='*', 
                   edgecolors='black', label='Base')
        
        plt.title(f'UAV Patrol Paths (Coverage: {self.coverage:.1%})')
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.grid(True)
        plt.legend()
        
        # Set axis limits
        if area_size:
            plt.xlim(0, area_size)
            plt.ylim(0, area_size)
        else:
            # Auto-calculate limits
            all_points = self.targets + [self.base]
            min_x = min(p.x for p in all_points)
            max_x = max(p.x for p in all_points)
            min_y = min(p.y for p in all_points)
            max_y = max(p.y for p in all_points)
            margin = max((max_x - min_x), (max_y - min_y)) * 0.1
            plt.xlim(min_x - margin, max_x + margin)
            plt.ylim(min_y - margin, max_y + margin)
            
        plt.show()
    
    def get_results(self):
        """
        Get planning results
        
        Returns:
            dict: Dictionary containing paths, mission time, coverage, etc.
        """
        return {
            "paths": self.paths,
            "task_time": self.task_time,
            "coverage": self.coverage,
            "covered_weight": self.total_weight,
            "all_weight": sum(t.weight for t in self.targets)
        }
    
    def print_summary(self):
        """Print planning summary to console (in Chinese)"""
        print("\n===== 路径规划结果摘要 =====")
        print(f"目标点总数: {len(self.targets)}")
        print(f"无人机数量: {len(self.paths)}")
        print(f"总覆盖权重: {self.total_weight:.2f}/{sum(t.weight for t in self.targets):.2f}")
        print(f"覆盖率: {self.coverage:.2%}")
        print(f"任务总时间: {self.task_time:.2f} 分钟")
        
        # Print path information for each UAV
        for i, path in enumerate(self.paths):
            print(f"\n无人机 {i+1} 路径:")
            path_points = []
            for p in path:
                if p.idx == -1:
                    path_points.append("基地")
                else:
                    path_points.append(f"目标{p.idx}(权重:{p.weight})")
            print(" → ".join(path_points))

    def load_targets_from_file(planner, file_path):
        """
        Load targets from a txt file and add them to the planner.

        Parameters:
            planner (TargetPlanner): The planner object
            file_path (str): Path to the txt file
        """
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                x = float(parts[0])
                y = float(parts[1])
                weight = float(parts[2]) if len(parts) >= 3 else 1
                planner.add_target(x, y, weight)