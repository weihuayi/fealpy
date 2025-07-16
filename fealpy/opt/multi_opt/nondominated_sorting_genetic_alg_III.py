from fealpy.backend import backend_manager as bm
from ...typing import TensorLike

class NondominatedSortingGeneticAlgIII():
    """
    NSGA-III (Non-dominated Sorting Genetic Algorithm III) for multi-objective optimization.

    This implementation specializes in handling pre-defined reference points and initial populations,
    extending the standard NSGA-III algorithm for customized optimization scenarios.

    Parameters:
        M (int): Number of objectives (≥ 2).
        dim (int): Dimension of decision variables.
        lb (float): Lower bound for all decision variables.
        ub (float): Upper bound for all decision variables.
        zr (Tensor): Pre-computed reference points, shape (N_ref, M).
        x (Tensor): Initial population of decision variables, shape (N_pop, dim).
        fobj (callable): Objective function with signature f(x) → Tensor of shape (N_pop, M).

    Attributes:
        zr (Tensor): Reference points array, shape (N_ref, M).
        pop (dict): Population dictionary containing:
            - 'x': Decision variables, shape (N_pop, dim)
            - 'fitness': Objective values, shape (N_pop, M)
            - 'rank': Non-dominated front numbers
            - 'distance_to_associated_ref': Distances to nearest reference points
        zmin (Tensor): Ideal point (best found objectives), shape (1, M).
        zmax (Tensor): Extreme points for normalization, shape (M, M).
        smin (Tensor): Minimum scalarization values, shape (1, M).
    """
    def __init__(
            self, 
            M: int, 
            dim: int, 
            lb: float, 
            ub: float, 
            zr: TensorLike,
            x: TensorLike,
            fobj: callable
        ):
        """
        Initialize NSGA-III with custom reference points and initial population.

        Parameters:
            M: Number of objectives.
            dim: Decision space dimension.
            lb: Lower bound for decision variables.
            ub: Upper bound for decision variables.
            zr: Pre-generated reference points, shape (N_ref, M).
            x: Initial population, shape (N_pop, dim).
            fobj: Objective function that computes fitness values.
        """
        self.pop = {}
        self.M = M
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.zr = zr
        self.pop['x'] = x
        self.pop['fitness'] = fobj(x)
        self.fun = fobj
        self.MaxIT = 400
        self.N = self.zr.shape[0]
        self.prev_zmin = bm.full((1, M), bm.inf)
        self.smin = bm.full((1, M), bm.inf)
        self.zmax = bm.zeros((M, M))
        self.sigma = 0.1 * (ub - lb)
    
    def update_ideal_point(self):
        """
        Update the ideal point (best found values for each objective).
        """
        zmin = bm.min(self.pop['fitness'], axis=0)
        self.zmin = bm.minimum(zmin, self.prev_zmin)

    def find_hyperplane_intercepts(self, fp):
        """
        Find intercepts for normalization hyperplane.

        Parameters:
            fp (Tensor): Population objective values, shape (n_pop, M).

        Returns:
            Tensor: Intercept vector, shape (1, M).
        """ 
        try:
            a = bm.linalg.solve(self.zmax.T, bm.ones((self.M, 1))).T
            if bm.any(a < 0):
                a = bm.max(fp, axis=0)
        except bm.linalg.LinAlgError:
            a = bm.max(fp, axis=0)
        return a

    def normalize_population(self):
        """
        Normalize objectives using ideal point and intercepts.

        Steps:
            1. Translate by ideal point: f' = f - zmin
            2. Compute intercepts a
            3. Normalize: f_norm = f' / a
        """
        self.update_ideal_point()
        fp = self.pop['fitness'] - self.zmin
        self.perform_scalarizing(fp)
        a = self.find_hyperplane_intercepts(fp)
        a = bm.minimum(a, 1e-10)
        self.pop['normalize_fitness'] = self.pop['fitness'] / a
    
    def perform_scalarizing(self, z):
        """
        Compute extreme points for normalization.

        Parameters:
            z (Tensor): Translated fitness values, shape (N_pop, M).

        Updates:
            zmax (Tensor): Extreme points, shape (M, M).
            smin (Tensor): Minimum scalarizing values, shape (1, M).
        """
        w = bm.ones((self.M, self.M)) * 1e-06
        w[bm.arange(self.M), bm.arange(self.M)] = 1
        s = bm.zeros((z.shape[0], self.M))
        for i in range(self.M):
            s[:, i] = bm.max(z / w[:, i], axis=1)
        sminj = bm.min(s, axis=0)
        ind = bm.argmin(s, axis=0)
        mask = sminj[None, :] < self.smin
        a = z[ind]
        self.zmax = bm.where(mask, z[ind], self.zmax)
        self.smin = bm.where(mask, sminj, self.smin)

    def non_dominated_sorting(self, pop):
        """
        Perform non-dominated sorting (Pareto ranking).

        Parameters:
            pop (dict): Population dictionary with 'fitness' key.

        Returns:
            dict: Updated population with 'rank' and domination sets.

        Complexity: O(M·N^2) where M=objectives, N=population size.
        """
        n = pop['x'].shape[0]
        pop['domination_set'] = [[] for _ in range(n)]
        pop['dominated_count'] = bm.zeros((n,), dtype=bm.int64)
        self.F = {'0': []}
        pop['rank'] = bm.zeros((n,), bm.int64)
        for i in range(n):
            p = pop['fitness'][i]
            for j in range(i+1, n):
                q = pop['fitness'][j]
                if self.dominate(p, q):
                    pop['dominated_count'][j] = pop['dominated_count'][j] + 1
                    pop['domination_set'][i].append(j)
                if self.dominate(q, p):
                    pop['dominated_count'][i] = pop['dominated_count'][i] + 1
                    pop['domination_set'][j].append(i)
            if pop['dominated_count'][i] == 0:
                self.F['0'].append(i)
                pop['rank'][i] = 0
        k = 0
        while True:
            Q = []
            for i in self.F[str(k)]:
                if not pop['domination_set'][i]:
                    continue
                for j in pop['domination_set'][i]:
                    pop['dominated_count'][j] = pop['dominated_count'][j] - 1
                    if pop['dominated_count'][j] == 0:
                        Q.append(j)
                        pop['rank'][j] = k + 1
            if not Q:
                break
            k += 1
            self.F[str(k)] = Q
        return pop

    def dominate(self, x, y):
        """
        Check Pareto dominance between solutions x and y.

        Parameters:
            x (Tensor): Objective values of first solution, shape (M,).
            y (Tensor): Objective values of second solution, shape (M,).

        Returns:
            bool: True if x dominates y (∀i x_i ≤ y_i ∧ ∃i x_i < y_i).
        """
        d = bm.all(x <= y) & bm.any(x < y)
        return d

    def associate_to_reference_point(self):
        """
        Associate solutions to nearest reference point.

        Returns:
            tuple: (d, rho) where
                - d: Distance matrix, shape (N_pop, N_ref)
                - rho: Reference point occupation counts, shape (N_ref,)
        """
        w =  self.zr / bm.linalg.norm(self.zr, axis=1)[:, None]
        dot_product = bm.matmul(self.pop['normalize_fitness'], w.T)
        proj = dot_product[:, :, None] * w[None, :, :]
        diff = self.pop['normalize_fitness'][:, None, :] - proj
        d = bm.linalg.norm(diff, axis=2) 
        self.pop['distance_to_associated_ref'] = bm.min(d, axis=1)
        self.pop['associated_ref'] = bm.argmin(d, axis=1)
        rho = bm.bincount(self.pop['associated_ref'], minlength=self.N)
        if rho.size < self.N:
            padding = bm.zeros(self.N - rho.size)
            rho = bm.concatenate([rho, padding])
        return d, rho

    def sort_and_select_population(self) -> None:
        """
        Select next generation using reference point-based niching.

        Procedure:
            1. Normalize objectives
            2. Non-dominated sort
            3. Associate solutions to reference points
            4. Select by rank and reference point density
        """
        self.normalize_population()
        self.pop = self.non_dominated_sorting(self.pop)
        if self.pop['x'].shape[0] == self.N:
            return
        d, rho = self.associate_to_reference_point() 
        new_pop = {}
        num = bm.array(list(map(len, self.F.values())))
        cum_num = bm.cumsum(num)
        if self.N in cum_num:
            index = bm.where(cum_num == self.N)
            arrays = [bm.array(self.F[str(i)]) for i in range(index[0][0]+1)]
            new_index = bm.concatenate(arrays)
            new_pop['x'] = self.pop['x'][new_index]
            new_pop['fitness'] = self.pop['fitness'][new_index]
            new_pop['normalize_fitness'] = self.pop['normalize_fitness'][new_index]
            new_pop['domination_set'] = [self.pop['domination_set'][i] for i in new_index]
            new_pop['dominated_count'] = self.pop['dominated_count'][new_index]
            new_pop['rank'] = self.pop['rank'][new_index]
            new_pop['distance_to_associated_ref'] = self.pop['distance_to_associated_ref'][new_index]
            new_pop['associated_ref'] = self.pop['associated_ref'][new_index]
        else:
            mask = cum_num < self.N
            index = bm.where(mask == True)
            a = index[0]
            if a.shape[0] == 0:
                key_f = bm.array(self.F['0'])
                while self.N not in cum_num:
                    j = bm.argmin(rho)
                    mask = self.pop['associated_ref'][key_f] == j
                    associted_from_last_front = key_f[mask]
                    if associted_from_last_front.size == 0:
                        rho[j] = 1e9
                        continue
                    if rho[j] == 0:
                        ddj = d[associted_from_last_front, j]
                        new_index = bm.argmin(ddj)
                    else:
                        new_index = bm.random.randint(0, len(associted_from_last_front), (1,))
                    member_to_add = associted_from_last_front[new_index]
                    key_f = key_f[key_f != member_to_add]
                    new_pop['x'] = self.pop['x'][member_to_add]
                    new_pop['fitness'] = self.pop['fitness'][member_to_add]
                    new_pop['normalize_fitness'] = self.pop['normalize_fitness'][member_to_add]
                    new_pop['domination_set'] = [self.pop['domination_set'][member_to_add.item()]]
                    new_pop['dominated_count'] = self.pop['dominated_count'][member_to_add]
                    new_pop['rank'] = self.pop['rank'][member_to_add]
                    new_pop['distance_to_associated_ref'] = self.pop['distance_to_associated_ref'][member_to_add]
                    new_pop['associated_ref'] = self.pop['associated_ref'][member_to_add]
                    
                    rho[j] = rho[j] + 1
                    break
            else:    
                a = index[-1]
                arrays = [bm.array(self.F[str(i)]) for i in range(a[-1]+1)]
                new_index = bm.concatenate(arrays)
                new_pop['x'] = self.pop['x'][new_index]
                key_f = bm.array(self.F[str(a[-1]+1)])
                new_pop['fitness'] = self.pop['fitness'][new_index]
                new_pop['rank'] = self.pop['rank'][new_index]
                new_pop['domination_set'] = [self.pop['domination_set'][i] for i in new_index]
                new_pop['dominated_count'] = self.pop['dominated_count'][new_index]
                new_pop['normalize_fitness'] = self.pop['normalize_fitness'][new_index]
                new_pop['associated_ref'] = self.pop['associated_ref'][new_index]
                new_pop['distance_to_associated_ref'] = self.pop['distance_to_associated_ref'][new_index]

        while self.N not in cum_num:
            j = bm.argmin(rho)
            mask = self.pop['associated_ref'][key_f] == j
            associted_from_last_front = key_f[mask]
            if associted_from_last_front.size == 0:
                rho[j] = 1e9
                continue
            if rho[j] == 0:
                ddj = d[associted_from_last_front, j]
                new_index = bm.argmin(ddj)
            else:
                new_index = bm.random.randint(0, len(associted_from_last_front), (1,))
            member_to_add = associted_from_last_front[new_index]
            key_f = key_f[key_f != member_to_add]
            new_pop['x'] = bm.concatenate([new_pop['x'], self.pop['x'][member_to_add]])
            new_pop['fitness'] = bm.concatenate([new_pop['fitness'], self.pop['fitness'][member_to_add]])
            new_pop['rank'] = bm.concatenate([new_pop['rank'], self.pop['rank'][member_to_add]])
            new_pop['domination_set'] = new_pop['domination_set'] + [self.pop['domination_set'][i] for i in member_to_add]
            new_pop['normalize_fitness'] = bm.concatenate(
                [
                    new_pop['normalize_fitness'], 
                    self.pop['normalize_fitness'][member_to_add]
                ]
            )
            new_pop['associated_ref'] = bm.concatenate(
                [
                    new_pop['associated_ref'], 
                    self.pop['associated_ref'][member_to_add]
                ]
            )
            new_pop['distance_to_associated_ref'] = bm.concatenate(
                [
                    new_pop['distance_to_associated_ref'], 
                    self.pop['distance_to_associated_ref'][member_to_add]
                ]
            )
            rho[j] = rho[j] + 1
            if new_pop['x'].shape[0] >= self.N:
                break

        self.pop = self.non_dominated_sorting(new_pop)
        pass

    def crossover(self):
        """
        Perform simulated binary crossover (SBX).

        Returns:
            tuple: (x_c, f_c) where
                - x_c: Offspring decision variables, shape (2*N_c, dim)
                - f_c: Offspring objective values, shape (2*N_c, M)
        """
        p1 = bm.random.randint(0, self.N, (self.N_c,))
        p2 = bm.random.randint(0, self.N, (self.N_c,))
        
        alpha = bm.random.rand(self.N_c, 1)
        x_c1 = alpha * self.pop['x'][p1] + (1 - alpha) * self.pop['x'][p2] 
        x_c2 = alpha * self.pop['x'][p2] + (1 - alpha) * self.pop['x'][p1] 
        x_c = bm.concatenate([x_c1, x_c2], axis=0)
        x_c = bm.clip(x_c, self.lb, self.ub)
        f_c = self.fun(x_c)
        return x_c, f_c

    def mutation(self):
        """
        Perform polynomial mutation.

        Returns:
            tuple: (x_m, f_m) where
                - x_m: Mutated decision variables, shape (N_m, dim)
                - f_m: Mutated objective values, shape (N_m, M)
        """
        m1 = bm.random.randint(0, self.N, (self.N_m,))
        x_m = self.pop['x'][m1].copy()
        j_m = bm.random.randint(0, self.dim, (self.N_m,))
        noise = bm.random.randn(self.N_m) * self.sigma
        x_m[bm.arange(self.N_m), j_m] += noise
        x_m = bm.clip(x_m, self.lb, self.ub)
        f_m = self.fun(x_m)
        return x_m, f_m

    def run(self, params={'pc':0.5, 'pm': 0.5, 'mu':0.02}):
        """
        Execute the NSGA-III optimization process.

        Parameters:
            params (dict): Algorithm parameters with keys:
                - pc: Crossover probability
                - pm: Mutation probability
                - mu: Mutation strength

        Returns:
            dict: Final population dictionary with optimized solutions.
        """
        self.mu = params.get('mu')
        self.pc = params.get('pc')
        self.pm = params.get('pm')
        self.N_c = int(bm.round(self.pc * self.N / 2))
        self.N_m = int(bm.round(self.N * self.pm))
        self.n_m = int(bm.ceil(self.mu * self.dim))
        self.sort_and_select_population()
        for it in range(self.MaxIT):
            print(it, self.F['0'].__len__())
            x_c, f_c = self.crossover()
            x_m, f_m = self.mutation()
            self.pop['x'] = bm.concatenate([self.pop['x'], x_c, x_m], axis=0)
            self.pop['fitness'] = bm.concatenate((self.pop['fitness'], f_c, f_m), axis=0)
            self.sort_and_select_population()
        return self.pop

if __name__ == "__main__":
    def DTLZ1(x, M, V):
        f = bm.zeros((x.shape[0], M))
        xm = V - M + 1
        gx = 100 * (
            xm + 
            bm.sum((x[:, M-1:] - 0.5) ** 2, axis=-1) - 
            bm.sum(bm.cos(20*bm.pi*(x[:, M-1:]-0.5)), axis=1)
        )
        f[:, 0] = 0.5 * bm.prod(x[:, :M-1], axis=1) * (1 + gx)
        for i in range(1, M-1):
            f[:, i] = 0.5 * bm.prod(x[:, :M-i-1], axis=-1) * (1 - x[:, M-i-1]) * (1 + gx)
        f[:, M-1] = 0.5 * (1 + gx) * (1 - x[:, 0])
        return f
    
    fobj = lambda x: DTLZ1(x, 3, 7)

    test = NondominatedSortingGeneticAlgIII(M=3, V=7, dim=7, lb=0, ub=1, H1=3, H2=2, fobj=fobj)
    pop = test.run()
    pass