import matplotlib.pyplot as plt

from ....backend import backend_manager as bm
from ....opt import *
from .travelling_salesman_prob import TravellingSalesmanProb

class MultipleTravelingSalesmanProb(TravellingSalesmanProb):
    """
    Multiple Traveling Salesman Problem (mTSP) / Multi-UAV routing problem
    with a two-level optimization structure.

    This class implements a bilevel optimization framework:
        - Upper-level: Assign cities (targets) to UAVs using a random-key encoding
        - Lower-level: Optimize the visiting order (route) for each UAV using 2-opt

    The upper-level objective minimizes the maximum tour length among all UAVs,
    leading to a balanced workload distribution.

    Features:
        - Fully vectorized batch evaluation for lower-level routing
        - Backend-agnostic (NumPy / PyTorch / JAX via FEALPy backend_manager)
        - Supports both single-route and batched 2-opt optimization

    Inheritance:
        Extends TravellingSalesmanProb by adding multi-route and assignment logic.
    """

    def __init__(self, options):
        """
        Initialize the mTSP / Multi-UAV routing problem.

        Parameters:
            options (dict): Configuration dictionary containing:
                - pos (Tensor): Coordinates of target cities, shape (N, 2)
                - warehouse (Tensor): Depot coordinate, shape (1, 2)
                - uav_num (int): Number of UAVs
                - up_opt_dict (dict): Upper-level optimizer configuration
        """

        self.options = options

        # concatenate city coordinates and depot
        pos = options['pos']
        warehouse = options['warehouse']
        self.pos = bm.concatenate([pos, warehouse])

        # full distance matrix (including depot)
        self.D = self._compute_distance_matrix(self.pos)

        # number of UAVs
        self.uav_num = options['uav_num']

        # upper-level optimization parameters
        self.up_alg = options['up_opt_dict']['opt_alg']
        self.up_N = options['up_opt_dict']['NP']
        self.up_T = options['up_opt_dict']['MaxIT']


    def cost_function(self, x):
        """
        Upper-level objective function.

        This function:
            1) Decodes random-key encoding into UAV-city assignments
            2) Constructs initial routes
            3) Optimizes each route via lower-level objective
            4) Returns the maximum UAV tour length

        Parameters:
            x (Tensor): Random-key encoding, shape (NP, city_num)

        Returns:
            Tensor: Upper-level fitness (minimize max UAV distance)
        """
        self.create_paths(x)
        fit = self.opt_paths()
        return fit

    def _evaluate_one_individual(self, i):
        """
        Evaluate a single upper-level individual.

        For the i-th individual:
            - Extract routes for all UAVs
            - Generate random initial permutations
            - Optimize routes using batched 2-opt
            - Return per-UAV route lengths

        Parameters:
            i (int): Index of the individual in the population

        Returns:
            Tensor: Route lengths for each UAV, shape (uav_num,)
        """
        counts = self._counts[i]        # (uav_num,)
        valid = counts > 0              # (uav_num,)
        u = self.uav_num

        # If no UAV has assigned targets, return zero cost
        if not bm.any(valid):
            return bm.zeros((u,))

        # Route length = number of cities + depot
        lens = counts + 1
        L_max = bm.max(lens)

        # Construct route validity mask
        ar = bm.arange(L_max)
        route_mask = ar[None, :] < lens[:, None]

        # Initialize batched distance matrices
        D_batch = bm.zeros((u, L_max, L_max))

        # Extract valid sub-matrices from precomputed D_arrays
        L = self.D_arrays.shape[-1]
        counts_i = self._counts[i]
        idx = bm.arange(L)[None, :]
        counts_mask = idx <= counts_i[:, None]
        mask2d = counts_mask[:, :, None] & counts_mask[:, None, :]
        D_big = self.D_arrays[i]

        # Mask invalid entries and truncate to L_max
        D_batch = bm.where(mask2d, D_big, 0)
        D_batch = D_batch[:, :L_max, :L_max]

        # ---------- 4) Generate batched initial routes x0 ----------
        mask = counts > 0
        routes = bm.zeros((u, L_max), dtype=bm.int64)
        n_row = u
        n_col = L_max

        # Base index array [0, 1, ..., L_max-1]
        idx = bm.arange(n_col)[None, :]            # (1, L_max)
        idx = bm.repeat(idx, n_row, axis=0)        # (u, L_max)

        # Valid positions: indices <= counts[i]
        mask = idx <= counts[:, None]

        # Random keys for permutation generation
        rand_key = bm.random.rand(n_row, n_col)

        # Assign large values to invalid positions to exclude them from sorting
        rand_key = bm.where(mask, rand_key, bm.inf)

        # argsort generates random permutations per UAV
        routes = bm.argsort(rand_key, axis=1)

        # Reset invalid positions to zero (or another padding value if preferred)
        routes = bm.where(mask, routes, 0)

        # ---------- 5) Evaluate batched lower-level objective ----------
        # Note: down_fobj_batch must support batched inputs

        f_batch = self.down_fobj_batch(routes, D_batch, route_mask)

        # ---------- 6) Output per-UAV objective values ----------
        f_out = bm.zeros((u,), dtype=f_batch.dtype)
        f_out[valid] = f_batch[valid]
        return f_out

    def down_fobj(self, routes, D_batch, route_mask):
        """
        Lower-level objective (non-batched).

        Applies classical 2-opt to each UAV route independently.

        Returns:
            Tensor: Route length for each UAV
        """
        u = routes.shape[0]
        f = bm.zeros((u,))

        for j in range(u):
            # valid length of current path
            valid_len = int(bm.sum(route_mask[j]))

            if valid_len <= 1:
                f[j] = 0
                continue

            # extract valid route and D
            r_j = routes[j, :valid_len]
            D_j = D_batch[j, :valid_len, :valid_len]

            # call your original 2-opt
            r_opt = self.two_opt(r_j, D_j)

            # compute closed-loop distance
            # D_j[r_opt[t], r_opt[t+1]]
            edges = D_j[r_opt[:-1], r_opt[1:]]
            cost = bm.sum(edges) + D_j[r_opt[-1], r_opt[0]]

            f[j] = cost
        return f

    def batch_closed_loop_cost(self, routes, D_batch, route_mask):
        u, L = routes.shape
        n = bm.sum(route_mask, axis=1)

        # next index (cyclic)
        idx = bm.arange(L)
        next_idx = (idx + 1) % L

        # gather edges
        batch = bm.arange(u)[:, None]

        a = routes
        b = routes[:, next_idx]

        dist = D_batch[batch, a, b]

        # mask invalid edges
        edge_mask = idx[None, :] < n[:, None]
        dist = bm.where(edge_mask, dist, 0)

        return bm.sum(dist, axis=1)

    def down_fobj_batch(self, routes, D_batch, route_mask):
        """
        Batched lower-level objective.

        Applies batched 2-opt and returns optimized route costs.

        Returns:
            Tensor: Optimized route costs for all UAVs
        """
        _, f = self.two_opt_batch(routes, D_batch, route_mask)
        return f

    def two_opt_batch(self, routes, D_batch, route_mask, alpha=10, n_trials=5):
        """
        Batched 2-opt local search for multiple UAV routes.

        Parameters:
            routes (Tensor): Initial routes, shape (uav_num, L)
            D_batch (Tensor): Distance matrices, shape (uav_num, L, L)
            route_mask (Tensor): Boolean mask indicating valid route positions
            alpha (int): Scaling factor for iteration budget
            n_trials (int): Number of random trials per route

        Returns:
            routes (Tensor): Optimized routes after batched 2-opt
            cost (Tensor): Corresponding closed-loop route costs
        """
        routes = bm.copy(routes)
        u, L = routes.shape
        valid_len = bm.sum(route_mask, axis=1)

        # current closed-loop cost for each UAV route
        cost = self.batch_closed_loop_cost(routes, D_batch, route_mask)

        # only routes with at least 4 nodes can be improved by 2-opt
        can_opt = valid_len >= 4
        if not bm.any(can_opt):
            return routes, cost

        for _ in range(n_trials * L):

            # --- 1. Randomly sample swap indices (i, j) in batch mode ---
            i = bm.zeros((u,), dtype=bm.int64)
            j = bm.zeros((u,), dtype=bm.int64)

            safe = valid_len >= 4
            if not bm.any(safe):
                return routes, cost

            # upper bounds for i and j per route
            hi_i = bm.maximum(valid_len - 2, bm.array(1))
            hi_j = bm.maximum(valid_len - 1, bm.array(2))

            # global upper bounds for random sampling
            hi_i_max = bm.max(hi_i)
            hi_j_max = bm.max(hi_j)

            # sample candidate indices
            tmp_i = bm.random.randint(1, hi_i_max, size=(u,))
            tmp_j = bm.random.randint(2, hi_j_max, size=(u,))

            # enforce per-route bounds
            tmp_i = bm.where(tmp_i < hi_i, tmp_i, 1)
            tmp_j = bm.where(tmp_j < hi_j, tmp_j, 2)
            tmp_j = bm.maximum(tmp_j, tmp_i + 1)

            # mask invalid routes
            i = bm.where(safe, tmp_i, 0)
            j = bm.where(safe, tmp_j, 0)

            # --- 2. Gather affected edges and compute cost difference ---
            batch = bm.arange(u)

            a = routes[batch, i - 1]
            b = routes[batch, i]
            c = routes[batch, j]
            d = routes[batch, j + 1]

            # 2-opt cost delta
            delta = (
                D_batch[batch, a, c]
            + D_batch[batch, b, d]
            - D_batch[batch, a, b]
            - D_batch[batch, c, d]
            )

            improve = (delta < 0) & can_opt

            if not bm.any(improve):
                continue

            # --- 3. Apply segment reversal only to improving UAV routes ---
            routes = self.batch_segment_reverse(routes, i, j, mask=improve)
            cost = cost + bm.where(improve, delta, 0.0)

        return routes, cost

    def two_opt(self, route, D, alpha=10):
        """
        Perform classical 2-opt local search on a single TSP route.

        This method iteratively applies 2-opt edge exchanges to improve a given
        closed tour by reversing route segments that yield a negative cost delta.
        The search terminates early if no improving move can be found.

        Typical use cases include:
            - Correctness validation and debugging
            - Final route refinement after batched or heuristic optimization

        Parameters:
            route (Tensor): A 1D tensor representing a TSP tour (node indices).
            D (Tensor): Distance matrix where D[i, j] denotes the cost from node i to j.
            alpha (int, optional): Scaling factor controlling the maximum number of
                search iterations (max_pass = alpha * route length).
                Defaults to 10.

        Returns:
            Tensor: The locally optimized route after applying 2-opt moves.
        """
        route = bm.copy(route)
        n = route.shape[0]
        max_pass = alpha * n

        if n <= 3:
            return route  # trivial

        idx = bm.arange(n)

        for _ in range(max_pass):
            # build i and j candidate grids
            I = idx[1:n-1][:, None]   # shape (n-2,1) values 1..n-2
            J = idx[2:n][None, :]     # shape (1,n-2) values 2..n-1

            mask = (J > I)            # upper-triangular valid pairs

            # gather node indices for the formula
            a = route[I - 1]          # shape (n-2,1)
            b = route[I]              # shape (n-2,1)
            c = route[J]              # shape (1,n-2)
            d = route[(J + 1) % n]    # shape (1,n-2), wrap for last->first

            # compute delta matrix (broadcasting should produce (n-2, n-2))
            delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]

            # mask invalid pairs by +inf
            delta = bm.where(mask, delta, bm.inf)

            # best improvement
            min_delta = bm.min(delta)

            if min_delta >= 0:
                break

            # find flat position of min
            pos = bm.argmin(delta)

            rows = delta.shape[0]
            cols = delta.shape[1]
            r = pos // rows
            cpos = pos % cols

            # recover actual i,j values
            i = r + 1   # because I = idx[1:n-1], index r -> value r+1
            j = cpos + 2  # because J = idx[2:n], index cpos -> value cpos+2

            # perform inclusive reversal of [i, j]
            seg = bm.flip(route[i:(j + 1)], axis=0)
            route[i:(j + 1)] = seg
        return route

    def opt_paths(self):
        """
        Evaluate all upper-level individuals by optimizing their corresponding paths.

        For each upper-level individual, this method:
            - Solves the lower-level routing problem for all UAVs
            - Collects per-UAV route costs
            - Uses the maximum route cost as the fitness value

        This formulation corresponds to a min–max objective commonly used in
        multi-UAV routing problems to balance workload or completion time.

        Returns:
            Tensor: Fitness values for all upper-level individuals,
            shape (up_N,).
        """
        fit = bm.zeros((self.up_N,))
        for i in range(self.up_N):
            f = self._evaluate_one_individual(i)
            fit[i] = bm.max(f)
        return fit

    def batch_segment_reverse(self, routes, i, j, mask=None):
        """
        Reverse route segments in batch form for selected indices.

        For each active batch index b, this method reverses the sub-sequence
        routes[b, i[b] : j[b] + 1]. The operation is typically used as a core
        update step in batched 2-opt local search.

        Parameters:
            routes (Tensor): Batched routes, shape (batch_size, L).
            i (Tensor): Start indices of segments to be reversed, shape (batch_size,).
            j (Tensor): End indices of segments to be reversed, shape (batch_size,).
            mask (Tensor | None, optional): Boolean mask indicating which batch
                elements are active. If None, all pairs with i < j are processed.

        Returns:
            Tensor: A new tensor with specified route segments reversed.
        """
        routes = bm.copy(routes)
        u = routes.shape[0]

        if mask is None:
            active = bm.nonzero(i < j)[0]
        else:
            active = bm.nonzero(mask & (i < j))[0]

        for b in active:
            routes[b, i[b]:j[b] + 1] = bm.flip(
                routes[b, i[b]:j[b] + 1],
                axis=0
            )

        return routes

    def create_paths(self, x):
        """
        Creates initial paths by assigning targets to UAVs based on solution encoding.

        Parameters:
            x (Tensor): Solution encoding representing target assignments.
        """

        up_N, city_num = x.shape
        u = self.uav_num
        divs = self.set_divs()                     # (u,)
        assigned = bm.searchsorted(divs, x)        # (up_N, city_num)

        depot = self.pos.shape[0] - 1

        # ========== 1) Build assignment mask ==========
        # mask[n, j, c] = True if city c is assigned to UAV j in solution n
        mask = (assigned[:, None, :] == bm.arange(u)[None, :, None])   # (up_N, u, city_num)

        # ========== 2) Extract city indices in ascending order (fully equivalent to non-vectorized version) ==========
        # For each solution n and UAV j:
        # rr = where(mask == True), which naturally follows city index order (0..city_num-1)
        city_idx = bm.arange(city_num)[None, None, :]      # (1, 1, city_num)

        # Replace unassigned cities with a large sentinel value
        invalid = city_num + 9999
        masked_city_idx = bm.where(mask, city_idx, invalid)     # (up_N, u, city_num)

        # count[n, j] = number of cities assigned to UAV j
        counts = bm.sum(mask, axis=2)                      # (up_N, u)

        max_len = int(bm.max(counts)) if up_N > 0 else 0

        # ========== 3) Collect ordered city indices per UAV and pad to equal length ==========
        # Unassigned cities are pushed to the end using argsort
        order_idx = bm.argsort(masked_city_idx, axis=2)    # (up_N, u, city_num)

        # Gather sorted city indices
        rr_sorted = masked_city_idx[
            bm.arange(up_N)[:, None, None],
            bm.arange(u)[None, :, None],
            order_idx
        ]  # (up_N, u, city_num)

        # The first counts[n,j] entries are valid cities; remaining are invalid
        rr_padded = rr_sorted[:, :, :max_len]              # (up_N, u, max_len)

        # Replace invalid entries with depot index to avoid distance pollution
        rr_valid = bm.where(rr_padded < invalid, rr_padded, depot)

        # ========== 4) Append depot to each route ==========
        depot_col = depot * bm.ones((up_N, u, 1), dtype=rr_valid.dtype)
        routes = bm.concatenate((rr_valid, depot_col), axis=2)   # (up_N, u, max_len + 1)

        # Store routes (structure differs from original implementation, but content is equivalent)
        self.salesman_path = routes

        # ========== 5) Construct batched distance matrices ==========
        L = max_len + 1
        idx_i = routes[..., :, None]   # (up_N, u, L, 1)
        idx_j = routes[..., None, :]   # (up_N, u, 1, L)
        self.D_arrays = self.D[idx_i, idx_j]    # (up_N, u, L, L)
        self._counts = counts

    def construct_D(self, r):
        """
        Constructs distance matrix for a given route.

        Parameters:
            r (Tensor): Indices of nodes in the route.

        Returns:
            Tensor: Distance matrix for the route.
        """
        return self.D[r[:, None], r[None, :]]

    def set_divs(self):
        """
        Sets division points for target assignment to UAVs.

        Returns:
            Tensor: Division points in [0,1] range for assignment.
        """
        divs = (bm.arange(self.uav_num) + 1) / self.uav_num
        return divs

    def solver(self):
        """
        Main solver method that runs the two-level optimization.
        """
        fobj = lambda x: self.cost_function(x)
        NP = self.up_N
        MaxIT = self.up_T
        dim = self.pos.shape[0] - 1
        x0 = initialize(NP, dim, 1, 0)
        option = opt_alg_options(x0, fobj, (0, 1), NP, MaxIters=MaxIT)
        self.up_optimizer = self.up_alg(option)
        self.up_optimizer.run()

    def visualization(self):
        """
        Visualize the final UAV routes stored in self.solution.
        """

        if not hasattr(self, "solution"):
            raise RuntimeError("No solution found. Please run output_route() first.")

        sol = self.solution

        # scatter all cities
        plt.scatter(self.pos[:, 0], self.pos[:, 1], color='red', zorder=3)

        # for each UAV, plot its route
        for key, info in sol.items():
            route = info["route"]
            if len(route) == 0:
                continue

            # Convert to numpy indices if backend is torch
            route_np = bm.asarray(route)

            # close loop for drawing
            route_np = bm.concatenate([route_np, route_np[:1]])

            plt.plot(
                self.pos[route_np, 0],
                self.pos[route_np, 1],
                label=f"{key}  (dist={info['distance']:.2f})",
                linewidth=1.8
            )

        plt.legend()
        plt.title("UAV Multi-Route Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def output_route(self, x):
        """
        Build and output the optimized route for each UAV.

        Parameters:
            x (Tensor): Final encoded random-key vector from the upper-level optimizer.
        """

        # Structured storage for all UAVs
        solution = {}

        # Step 1: determine assignment by random-key thresholding
        divs = self.set_divs()
        assigned = bm.searchsorted(divs, x)

        # Step 2: iterate over UAVs
        for u in range(self.uav_num):

            # ---- cities assigned to this UAV ----
            mask = (assigned == u)
            rr = bm.nonzero(mask)[0]

            # If no city assigned → skip or handle
            if rr.shape[0] == 0:
                solution[f"uav_{u}"] = {
                    "route": [],
                    "distance": 0.0
                }
                continue

            # Append depot (last index = D.shape[0]-1)
            depot = self.D.shape[0] - 1
            path = bm.concatenate((rr, bm.array([depot])))

            # Build sub-distance matrix
            D_sub = self.construct_D(path)

            # ---- Step 3: get initial permutation ----
            n_sub = D_sub.shape[0]

            if bm.backend_name == "numpy":
                route0 = bm.random.permutation(n_sub)
            elif bm.backend_name == "pytorch":
                route0 = bm.random.randperm(n_sub)
            else:
                raise RuntimeError("Unsupported backend")

            # ---- Step 4: apply 2-opt ----
            route = self.two_opt(route0, D_sub)

            # Close the loop (return to start)
            route = bm.concatenate((route, bm.array([route[0]])))

            # Map back to original city index
            route_global = path[route]

            # ---- Step 5: compute distance ----
            total_dist = bm.sum(self.D[route_global[:-1], route_global[1:]])

            # ---- Step 6: store in dictionary ----
            solution[f"uav_{u}"] = {
                "route": route_global.tolist(),
                "distance": float(total_dist)
            }

            # You may still print:
            print(f"UAV {u}: route = {route_global.tolist()}, distance = {float(total_dist)}")

        # store in object
        self.solution = solution