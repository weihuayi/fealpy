import itertools

from ..backend import backend_manager as bm
from .opt_function import initialize
from .optimizer_base import Optimizer
import matplotlib.pyplot as plt


class MO_QuantumParticleSwarmOpt(Optimizer):
    """
    MOQPSO: Multi-Objective Quantum-behaved Particle Swarm Optimization.

    This class implements a multi-objective version of QPSO, maintaining a repository 
    of non-dominated solutions and utilizing a grid-based leader selection mechanism.
    """
    def __init__(self, options) -> None:
        super().__init__(options)

    def run(self, params={'mut':0.5}):
        """
        Main optimization loop.
        """
        self.mut = params.get('mut')
        self.fitness = self.fun(self.x)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(self.fitness)
        
        # Initialize repository with non-dominated solutions
        Dominated = self.checkDomination(pbest_f)
        self.REP['pos'] = self.x[Dominated == 0, :]
        self.REP['fit'] = self.fitness[Dominated == 0, :]
        self.updateGrid()

        self.set_fit()
            
        self.plotting()
        print("Generation #0 - Repository size: ", self.REP['pos'].shape[0])

        # Main iteration loop
        for self.it in range(1, self.MaxIT):
            alpha = 0.9 - 0.5 * (self.it / self.MaxIT)
            mbest = bm.sum(pbest, axis=0) / self.N
            phi = bm.random.rand(self.N, self.dim)
            h = self.selectLeader()
            p = phi * pbest + (1 - phi) * self.REP['pos'][h]
            u = bm.random.rand(self.N, self.dim)
            
            # Update positions
            self.x = p + alpha * bm.abs(mbest - self.x) * bm.log(1 / u) * (1 - 2 * (bm.random.rand(self.N, 1) > 0.5))
            self.x = bm.clip(self.x, self.lb, self.ub)
            
            # Mutation
            self.mutation()
            
            # Fitness evaluation
            self.fitness = self.fun(self.x)
            
            # Update personal best
            pos_best = self.dominate(self.fitness, pbest_f)
            best_pos = 1 - self.dominate(pbest_f, self.fitness)
            mask = bm.random.rand(self.N) > 0.5
            best_pos[(mask == 1)] = 0
            if bm.sum(pos_best) > 1:
                pbest = self.x * (pos_best[:, None] == 1) + pbest * (pos_best[:, None] == 0)
                pbest_f = self.fitness * (pos_best[:, None] == 1) + pbest_f * (pos_best[:, None] == 0)
            if bm.sum(best_pos) > 1:
                pbest = self.x * (best_pos[:, None] == 1) + pbest * (best_pos[:, None] == 0)
                pbest_f = self.fitness * (best_pos[:, None] == 1) + pbest_f * (best_pos[:, None] == 0)
            
            # Update repository
            self.updateRepository()
            if self.REP['pos'].shape[0] > self.Nr:
                self.deleteFromRepository()
            self.plotting()
            print("Generation #", self.it, "- Repository size: ", self.REP['pos'].shape[0])

        plt.ioff()
        plt.show()

    def set_fit(self):
        if self.REP['fit'].shape[1] == 2:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        elif self.REP['fit'].shape[1] == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

    def deleteFromRepository(self):
        """
        Delete crowded solutions from the repository to maintain the repository size.
        """
        crowding = bm.zeros((self.REP['pos'].shape[0], 1))
        for m in range(self.REP['fit'].shape[1]):
            m_fit = bm.sort(self.REP['fit'][:, m])
            idx = bm.argsort(self.REP['fit'][:, m])
            m_up = bm.concatenate([m_fit[1:], bm.array([bm.inf])])
            m_down = bm.concatenate([bm.array([bm.inf]), m_fit[:-1]])
            distance = ((m_up - m_down) / (bm.max(m_fit) - bm.min(m_fit)))
            idx = bm.argsort(idx)
            distance[distance == -bm.inf] = bm.inf
            crowding = crowding + distance[idx][:, None]
        crowding[bm.isnan(crowding)] = bm.inf
        del_idx = bm.argsort(crowding, axis=0)
        n_del = self.REP['pos'].shape[0] - self.Nr
        del_idx = del_idx[:n_del]

        # Keep the remaining solutions
        j = 0
        pos = bm.zeros((self.Nr, self.dim))
        fit = bm.zeros((self.Nr, self.REP['fit'].shape[1]))
        for i in range(self.REP['pos'].shape[0]):
            if i not in del_idx:
                pos[j, :] = self.REP['pos'][i, :]
                fit[j, :] = self.REP['fit'][i, :]
                j += 1
        self.REP['pos'] = bm.copy(pos)
        self.REP['fit'] = bm.copy(fit)
        self.updateGrid()

    def updateRepository(self):
        """
        Update the repository with current non-dominated solutions.
        """
        Dominated = self.checkDomination(self.fitness)
        self.REP['pos'] = bm.concatenate((self.REP['pos'], self.x[Dominated == 0, :]))
        self.REP['fit'] = bm.concatenate((self.REP['fit'], self.fitness[Dominated == 0, :]))
        Dominated = self.checkDomination(self.REP['fit'])
        self.REP['pos'] = self.REP['pos'][Dominated == 0, :]
        self.REP['fit'] = self.REP['fit'][Dominated == 0, :]
        self.updateGrid()

    def mutation(self):
        """
        Apply mutation operations to enhance exploration.
        """
        fract = self.N / 3 - bm.floor(bm.array(self.N) / 3)
        if fract < 0.5:
            sub_sizes = bm.array([int(bm.ceil(bm.array(self.N) / 3)), int(bm.round(bm.array(self.N) / 3)), int(bm.round(bm.array(self.N) / 3))])
        else:
            sub_sizes = bm.array([int(bm.round(bm.array(self.N) / 3)), int(bm.round(bm.array(self.N) / 3)), int(bm.floor(bm.array(self.N) / 3))])
        cum_sizes = bm.cumsum(sub_sizes, axis=0)

        # First type of mutation
        nmut = bm.round(self.mut * sub_sizes[1])
        if nmut > 0:
            idx = cum_sizes[0] + bm.unique(bm.random.randint(0, sub_sizes[1], (int(nmut),)))
            self.x[idx] = initialize(idx.shape[0], self.dim, self.ub, self.lb)

        # Second type of mutation
        per_mut = (1 - self.it / self.MaxIT) ** (5 * self.dim)
        nmut = bm.round(per_mut * sub_sizes[2])
        if nmut > 0:
            idx = cum_sizes[1] + bm.unique(bm.random.randint(0, sub_sizes[2], (int(nmut),)))
            self.x[idx] = initialize(idx.shape[0], self.dim, self.ub, self.lb)

    def selectLeader(self):
        """
        Select a leader (guiding solution) from the repository based on crowding information.
        """
        prob = bm.cumsum(self.REP['quality'][:, 1], axis=0)
        rand_val = bm.random.rand(1) * bm.max(prob)
        selected_idx = bm.where(rand_val <= prob)[0][0]
        sel_hyp = self.REP['quality'][selected_idx, 0]
        idx = bm.arange(0, self.REP['grid_idx'].shape[0])[:, None]
        selected = idx[self.REP['grid_idx'] == sel_hyp]
        selected = selected[bm.random.randint(0, selected.shape[0], (1,))]
        return selected

    def plotting(self):
        """
        Plot the current population and repository (only works for bi-objective problems).
        """
        if self.REP['fit'].shape[1] == 2:
            plt.ion()
            self.ax.clear()
            plt.plot(self.fitness[:, 0], self.fitness[:, 1], 'o', markerfacecolor='none', markeredgecolor='red')
            plt.plot(self.REP['fit'][:, 0], self.REP['fit'][:, 1], 'o', markerfacecolor='none', markeredgecolor='black')
            plt.plot(self.PF[:, 0], self.PF[:, 1], '.', markeredgecolor='green')
            self.ax.grid(True)
            self.ax.set_ylim(bm.min(self.REP['fit'][:, 1]), bm.max(self.REP['fit'][:, 1]))
            self.ax.set_xlim(bm.min(self.REP['fit'][:, 0]), bm.max(self.REP['fit'][:, 0]))
            plt.gca().set_xticks(self.REP['hypercube_limits'][:, 0])
            plt.gca().set_yticks(self.REP['hypercube_limits'][:, 1])
            plt.xlabel('F1')
            plt.ylabel('F2')
            plt.show()
            plt.pause(0.01)
            plt.draw()
        
        if self.REP['fit'].shape[1] == 3:
            plt.ion()
            self.ax.clear()
            self.ax.set_xlabel('F1')
            self.ax.set_ylabel('F2')
            self.ax.set_zlabel('F3')
            self.ax.set_xlim(bm.min(self.REP['fit'][:, 0]), bm.max(self.REP['fit'][:, 0]))
            self.ax.set_ylim(bm.min(self.REP['fit'][:, 1]), bm.max(self.REP['fit'][:, 1]))
            self.ax.set_zlim(bm.min(self.REP['fit'][:, 2]), bm.max(self.REP['fit'][:, 2]))
            self.ax.plot(self.fitness[:, 0], self.fitness[:, 1], self.fitness[:, 2], 'o', markerfacecolor='none', markeredgecolor='red')
            self.ax.plot(self.REP['fit'][:, 0], self.REP['fit'][:, 1], self.REP['fit'][:, 2], 'o', markerfacecolor='none', markeredgecolor='black')
            self.ax.plot(self.PF[:, 0], self.PF[:, 1], self.PF[:, 2], '.', markeredgecolor='green')
            self.ax.set_xticks(self.REP['hypercube_limits'][:, 0])
            self.ax.set_yticks(self.REP['hypercube_limits'][:, 1])
            self.ax.set_zticks(self.REP['hypercube_limits'][:, 2])
            plt.show()
            plt.pause(0.01)
            plt.draw()


    def updateGrid(self):
        """
        Update the hypercube grid structure for selecting leaders.
        """
        ndim = self.REP['fit'].shape[1]
        self.REP['hypercube_limits'] = bm.zeros((self.ngrid + 1, ndim))
        for i in range(ndim):
            self.REP['hypercube_limits'][:, i] = bm.linspace(bm.min(self.REP['fit'][:, i]), bm.max(self.REP['fit'][:, i]), self.ngrid + 1)
        
        npar = self.REP['fit'].shape[0]
        self.REP['grid_idx'] = bm.zeros((npar, 1))
        self.REP['grid_subid'] = bm.zeros((npar, ndim))

        for n in range(npar):
            for d in range(ndim):
                condition = (self.REP['fit'][n, d] <= self.REP['hypercube_limits'][:, d]) * 1
                self.REP['grid_subid'][n, d] = bm.argmax(condition) if bm.any(condition) else None
                self.REP['grid_subid'][n, d] -= 1
                if self.REP['grid_subid'][n, d] == -1:
                    self.REP['grid_subid'][n, d] = 0
            self.REP['grid_idx'][n] = self.REP['grid_subid'][n, 1] * self.ngrid + self.REP['grid_subid'][n, 0]

        ids = bm.unique(self.REP['grid_idx'])
        self.REP['quality'] = bm.zeros((ids.shape[0], 2))
        for i in range(ids.shape[0]):
            self.REP['quality'][i, 0] = ids[i]
            self.REP['quality'][i, 1] = 10 / bm.sum(self.REP['grid_idx'] == ids[i])

    def checkDomination(self, fitness):
        """
        Check which solutions are dominated.

        Args:
            fitness (tensor): Fitness values.

        Returns:
            tensor: Domination vector (0 = non-dominated, 1 = dominated).
        """
        dom_vector = bm.zeros((fitness.shape[0],))
        all_perm = bm.array(list(itertools.combinations(range(1, fitness.shape[0] + 1), 2))) - 1
        a = bm.stack((all_perm[:, 1], all_perm[:, 0]), axis=1)
        all_perm = bm.concatenate((all_perm, a))
        d = self.dominate(fitness[all_perm[:, 0], :], fitness[all_perm[:, 1], :])
        dd = all_perm[d == 1, 1]
        dominated_particles = bm.unique(dd)
        dom_vector[dominated_particles] = 1
        return dom_vector

    def dominate(self, x, y):
        """
        Determine dominance between two solutions.

        Args:
            x (tensor): First set of solutions.
            y (tensor): Second set of solutions.

        Returns:
            tensor: Dominance result (1 = x dominates y, 0 = otherwise).
        """
        d = bm.all(x <= y, axis=1) & bm.any(x < y, axis=1)
        d = bm.where(d == True, 1, 0)
        return d


if __name__ == "__main__":
    from fealpy.opt.benchmark.multi_benchmark import multi_benchmark_data as data
    MultiObj = data[5]

    params = {}
    params['N'] = 200
    params['NR'] = 200
    params['MaxIT'] = 100
    params['mut'] = 0.5
    params['ngrid'] = 20
    
    test = MO_QuantumParticleSwarmOpt(MultiObj, params)
    test.run()
