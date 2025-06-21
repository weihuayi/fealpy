from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

"""
Hippopotamus Optimization Algorithm

Reference
~~~~~~~~~
Mohammad Hussein Amiri, Nastaran Mehrabi Hashjin, Mohsen Montazeri, Seyedali Mirjalili, Nima Khodadadi. 
Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm.
Scientific reports, 2024, 14: 5032.
"""

class HippopotamusOptAlg(Optimizer):
    """
    A class implementing the Hippopotamus Optimization Algorithm (HOA) for solving optimization problems.

    The Hippopotamus Optimization Algorithm is inspired by the behavior and movement patterns of hippopotamuses.
    It uses a combination of exploration and exploitation strategies to search for the optimal solution.

    Parameters:
        option (dict): A dictionary containing parameters for the optimizer such as population size, dimensionality, and iteration limits.

    Attributes:
        gbest (Tensor): The global best solution found by the algorithm.
        gbest_f (float): The fitness value of the global best solution.
        curve (Tensor): A tensor to store the progress of the best solution over iterations.

    Methods:
        run(): Main method to run the Hippopotamus Optimization Algorithm and update the population based on fitness values.

    Reference
        Mohammad Hussein Amiri, Nastaran Mehrabi Hashjin, Mohsen Montazeri, Seyedali Mirjalili, Nima Khodadadi. 
        Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm.
        Scientific reports, 2024, 14: 5032.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Hippopotamus Optimization algorithm with the provided options.

        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Hippopotamus Optimization Algorithm to find the optimal solution.

        This method iteratively updates the population based on exploration and exploitation strategies. 
        It uses various movement strategies influenced by fitness and random factors.

        The algorithm proceeds in several phases:
        1. **Exploration**: Hippopotamuses move toward the global best solution and adjust their positions.
        2. **Exploitation**: The population adjusts based on the local group mean and fitness differences.
        3. **Leader Selection and Group Dynamics**: Individuals interact with the leader and group based on fitness and random influences.
        4. **Boundary Handling**: Solutions are constrained within the defined boundaries.

        The global best solution and its fitness value are updated during each iteration.
        """
        # Initial fitness and global best solution
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx].reshape(1, -1)

        # Main loop for iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update hippopotamus positions

            # Temperature adjustment based on iteration
            T = bm.exp(bm.array(it / self.MaxIT))  # Eq.(5)
            i1 = bm.array(int(self.N / 2))
            I1 = bm.random.randint(1, 3, (i1, 1))
            I2 = bm.random.randint(1, 3, (i1, 1))

            # Movement coefficients (Alfa values) initialization
            Alfa0 = I2 * bm.random.rand(i1, self.dim) + bm.random.randint(0, 2, (i1, 1))
            Alfa1 = 2 * bm.random.rand(i1, self.dim) - 1
            Alfa2 = bm.random.rand(i1, self.dim)
            Alfa3 = I1 * bm.random.rand(i1, self.dim) + bm.random.randint(0, 2, (i1, 1))
            Alfa4 = bm.random.rand(i1, 1) * bm.ones((i1, self.dim))

            # Group dynamics initialization
            AA = bm.random.randint(0, 5, (i1,))[:, None]
            BB = bm.random.randint(0, 5, (i1,))[:, None]
            A = (AA == 0) * Alfa0 + (AA == 1) * Alfa1 + (AA == 2) * Alfa2 + (AA == 3) * Alfa3 + (AA == 4) * Alfa4
            B = (BB == 0) * Alfa0 + (BB == 1) * Alfa1 + (BB == 2) * Alfa2 + (BB == 3) * Alfa3 + (BB == 4) * Alfa4

            # Group selection and mean calculation
            RandGroupNumber = bm.random.randint(1, self.N + 1, (i1,))
            MeanGroup = bm.zeros((i1, self.dim))
            for i in range(i1):
                RandGroup = bm.unique(bm.random.randint(0, self.N - 1, (RandGroupNumber[i],)))
                MeanGroup[i] = self.x[RandGroup].mean(axis=0)

            # Update positions based on the global best
            X_P1 = self.x[: i1] + bm.random.rand(i1, 1) * (self.gbest - I1 * self.x[: i1])  # Eq.(3)
            X_P1 = X_P1 + (self.lb - X_P1) * (X_P1 < self.lb) + (self.ub - X_P1) * (X_P1 > self.ub)
            F_P1 = self.fun(X_P1)

            # Update positions based on fitness improvement
            mask = F_P1 < fit[: i1]
            self.x[: i1], fit[: i1] = bm.where(mask[:, None], X_P1, self.x[: i1]), bm.where(mask, F_P1, fit[: i1])

            # Exploitation phase: Adjust positions based on the group mean
            if T > 0.6:
                X_P2 = self.x[: i1] + A * (self.gbest - I2 * MeanGroup)  # Eq.(6)
            else:
                r2 = bm.random.rand(i1, 1)
                X_P2 = ((r2 > 0.5) * (self.x[: i1] + B * (MeanGroup - self.gbest)) + 
                        (r2 <= 0.5) * (self.lb + bm.random.rand(i1, 1) * (self.ub - self.lb)))

            # Apply boundary constraints
            X_P2 = X_P2 + (self.lb - X_P2) * (X_P2 < self.lb) + (self.ub - X_P2) * (X_P2 > self.ub)
            F_P2 = self.fun(X_P2)

            # Update positions based on fitness improvement
            mask = F_P2 < fit[: i1]
            self.x[: i1] = bm.where(mask[:, None], X_P2, self.x[: i1])
            fit[: i1] = bm.where(mask, F_P2, fit[: i1])

            # Predators' interaction with the population
            predator = self.lb + bm.random.rand(i1, self.dim) * (self.ub - self.lb)  # Eq.(10)
            F_HL = self.fun(predator)
            distance2Leader = abs(predator - self.x[i1:])  # Eq.(11)
            RL = 0.05 * levy(i1, self.dim, 1.5)  # Eq.(13)

            # Calculate new positions for predators
            X_P3 = ((fit[i1:] > F_HL)[:, None] * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / 
                     ((bm.random.rand(i1, 1) * 0.5 + 1) - (bm.random.rand(i1, 1) + 2) * 
                     bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / distance2Leader) + 
                    (fit[i1:] <= F_HL)[:, None] * 
                    (RL * predator + (bm.random.rand(i1, 1) * 2 + 2) / 
                     ((bm.random.rand(i1, 1) * 0.5 + 1) - (bm.random.rand(i1, 1) + 2) * 
                     bm.cos(2 * bm.pi * (bm.random.rand(i1, 1) * 2 - 1))) / 
                     (bm.random.rand(i1, self.dim) + 2 * distance2Leader)))

            # Apply boundary constraints
            X_P3 = X_P3 + (self.lb - X_P3) * (X_P3 < self.lb) + (self.ub - X_P3) * (X_P3 > self.ub)
            F_P3 = self.fun(X_P3)

            # Update positions based on fitness improvement
            mask = F_P3 < fit[: i1]
            self.x[: i1], fit[: i1] = bm.where(mask[:, None], X_P3, self.x[: i1]), bm.where(mask, F_P3, fit[: i1])

            # Local adjustment of bounds
            l_local = self.lb / (it + 1)
            h_local = self.ub / (it + 1)

            # Further adjustments with random factors
            Blfa0 = 2 * bm.random.rand(self.N, self.dim) - 1
            Blfa1 = bm.random.rand(self.N, 1) * bm.ones((self.N, self.dim))
            Blfa2 = bm.random.randn(self.N, 1) * bm.ones((self.N, self.dim))

            DD = bm.random.randint(0, 3, (self.N,))[:, None] 
            D = (DD == 0) * Blfa0 + (DD == 1) * Blfa1 + (DD == 2) * Blfa2

            X_P4 = self.x + bm.random.rand(self.N, 1) * (l_local + D * (h_local - l_local))  # Eq.(17)
            X_P4 = X_P4 + (self.lb - X_P4) * (X_P4 < self.lb) + (self.ub - X_P4) * (X_P4 > self.ub)
            F_P4 = self.fun(X_P4)

            # Update positions based on fitness improvement
            mask = F_P4 < fit
            self.x, fit = bm.where(mask[:, None], X_P4, self.x), bm.where(mask, F_P4, fit)

            # Update global best solution
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f
