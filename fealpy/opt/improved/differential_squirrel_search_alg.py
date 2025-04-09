from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class DifferentialSquirrelSearchAlg(Optimizer):
    """
    Differential Squirrel Search Algorithm (DSSA) for optimization.
    
    This class implements the Differential Squirrel Search Algorithm (DSSA),
    which is a population-based optimization technique inspired by the foraging
    behavior of squirrels. The algorithm performs optimization by updating the
    positions of squirrels based on the best solution found so far and differential
    evolution strategies.

    Attributes:
        gbest (bm.Tensor): The global best solution found so far.
        gbest_f (float): The fitness value of the global best solution.
        x (bm.Tensor): The population of candidate solutions.
        lb (bm.Tensor): The lower bounds for the variables.
        ub (bm.Tensor): The upper bounds for the variables.
        MaxIT (int): The maximum number of iterations for the algorithm.
        N (int): The size of the population.
        dim (int): The dimensionality of the problem.
        fun (function): The objective function to minimize.
        curve (bm.Tensor): The fitness values over iterations.

    Reference:
        Bibekananda Jena, Manoj Kumar Naik, Aneesh Wunnava, Rutuparna Panda.
        A Differential Squirrel Search Algorithm.
        In: Das, S., Mohanty, M.N. (eds) Advances in Intelligent Computing and Communication. Lecture Notes in Networks and Systems, vol 202. Springer, Singapore.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Differential Squirrel Search Algorithm with the given options.

        Args:
            option (dict): A dictionary containing parameters for the algorithm,
                           such as population size, bounds, and other configuration options.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Differential Squirrel Search Algorithm to optimize the objective function.
        
        The algorithm iterates through the population and updates the positions of the
        squirrels based on a combination of differential evolution and random search,
        ultimately minimizing the objective function.

        Steps:
            1. Evaluate the fitness of the initial population.
            2. Find the global best squirrel and the top 3 squirrels for differential evolution.
            3. Perform differential evolution and random search to update the population.
            4. Update the global best solution if a better solution is found.
            5. Track the fitness value of the best solution in each iteration.

        The process continues for the maximum number of iterations (MaxIT).
        """
        fit = self.fun(self.x)  # Evaluate the fitness of the current population.
        gbest_index = bm.argmin(fit)  # Find the index of the best solution.
        self.gbest_f = fit[gbest_index]  # Set the fitness of the global best solution.

        index = bm.argsort(fit)  # Sort the population by fitness.
        self.gbest = self.x[index[0]]  # Set the global best solution.
        FSa = self.x[index[1: 4]]  # Get the top 3 best solutions.
        FSa_fit = fit[index[1: 4]]
        FSn = self.x[index[4: self.N]]  # Get the remaining solutions.
        FSn_fit = fit[index[4: self.N]]
        Gc = 1.9  # Constant for differential evolution.
        Pdp = 0.1  # Probability for differential evolution.
        dg = 0.8  # Scaling factor for differential evolution.
        Cr = 0.5  # Crossover rate for differential evolution.

        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update the position of the squirrels based on differential evolution.

            n2 = bm.random.randint(4, self.N, (1,))
            index2 = bm.unique(bm.random.randint(0, self.N - 4, (n2[0],)))  # Random selection for squirrels.
            index3 = bm.array(list(set(bm.arange(self.N - 4).tolist()).difference(index2.tolist())))

            n2 = len(index2)
            n3 = self.N - n2 - 4

            # Differential evolution for the best 3 squirrels.
            r1 = bm.random.rand(3, 1)
            FSa_t = ((r1 > Pdp) * (FSa + dg * Gc * (self.gbest - FSa - bm.mean(self.x, axis=0))) + 
                     (r1 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(3, self.dim)))
            FSa_t = FSa_t + (self.lb - FSa_t) * (FSa_t < self.lb) + (self.ub - FSa_t) * (FSa_t > self.ub)
            mask = (bm.random.rand(3, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(3, self.dim)) == bm.arange(0, self.dim))
            Xa_t = bm.where(mask, FSa_t, FSa)
            mask = self.fun(Xa_t) < self.fun(FSa_t)
            FSa_t = bm.where(mask[:, None], Xa_t, FSa)

            # Differential evolution for the remaining squirrels.
            r2 = bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn_Xt = ((r2 > Pdp) * (FSn[index2] + dg * Gc * (FSa[t] - FSn[index2])) + 
                     (r2 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n2, self.dim)))
            FSn_Xt = FSn_Xt + (self.lb - FSn_Xt) * (FSn_Xt < self.lb) + (self.ub - FSn_Xt) * (FSn_Xt > self.ub)
            mask = (bm.random.rand(n2, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(n2, self.dim)) == bm.arange(0, self.dim))
            Xn_t = bm.where(mask, FSn_Xt, FSn[index2])
            mask = self.fun(Xn_t) < self.fun(FSn_Xt)
            FSn_Xt = bm.where(mask[:, None], Xn_t, FSn_Xt)

            # Additional differential evolution for the last set of squirrels.
            r3 = bm.random.rand(n3, 1)
            FSn_Yt = ((r3 > Pdp) * (FSn[index3] + dg * Gc * (self.gbest - FSn[index3])) + 
                      (r3 <= Pdp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n3, self.dim)))
            FSn_Yt = FSn_Yt + (self.lb - FSn_Yt) * (FSn_Yt < self.lb) + (self.ub - FSn_Yt) * (FSn_Yt > self.ub)
            mask = (bm.random.rand(n3, self.dim) < Cr) + (bm.round(self.dim * bm.random.rand(n3, self.dim)) == bm.arange(0, self.dim))
            Yn_t = bm.where(mask, FSn_Yt, FSn[index3])
            mask = self.fun(Yn_t) < self.fun(FSn_Yt)
            FSn_Yt = bm.where(mask[:, None], Yn_t, FSn_Yt)

            FSn_t = bm.concatenate((FSn_Xt, FSn_Yt), axis=0)

            # Update the global best solution.
            gbest_t = self.gbest + dg * Gc * (self.gbest - bm.mean(FSa, axis=0))
            gbest_t = gbest_t + (self.lb - gbest_t) * (gbest_t < self.lb) + (self.ub - gbest_t) * (gbest_t > self.ub)
            mask = self.fun(gbest_t[None, :]) < self.gbest_f
            self.gbest, self.gbest_f = bm.where(mask, gbest_t, self.gbest), bm.where(mask, self.fun(gbest_t[None, :]), self.gbest_f)

            # Combine the updated solutions and evaluate their fitness.
            FS_t = bm.concatenate((self.gbest[None, :], FSa_t, FSn_t), axis=0)
            Z_fit = self.fun(FS_t)
            mask = Z_fit < fit
            self.x = bm.where(mask[:, None], FS_t, self.x)
            fit = bm.where(mask, Z_fit, fit)

            # Sort the population and update the global best if necessary.
            index = bm.argsort(fit)
            gbest_mew = self.x[index[0]]
            gbest_f_mew = fit[index[0]]
            if gbest_f_mew <= self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew
            FSa = self.x[index[1: 4]]
            FSn = self.x[index[4: self.N]]
            self.curve[it] = self.gbest_f  # Record the best fitness value for this iteration.
