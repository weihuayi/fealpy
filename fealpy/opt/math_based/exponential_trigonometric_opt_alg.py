from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ExponentialTrigonometricOptAlg(Optimizer):
    """
    An Exponential Trigonometric Optimization Algorithm (ETOA), inheriting from the Optimizer class.

    This class implements the Exponential Trigonometric Optimization Algorithm, which combines exponential 
    and trigonometric functions to explore and exploit the search space effectively. It initializes with 
    a set of options and iteratively improves the solution through two phases: the first phase focuses 
    on exploration, while the second phase emphasizes exploitation.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(): Executes the Exponential Trigonometric Optimization Algorithm.

    Reference:
    ~~~~~~~~~~
    Tran Minh Luan, Samir Khatir, Minh Thi Tran, Bernard De Baets, Thanh Cuong-Le.
    Exponential-trigonometric optimization algorithm for solving complicated engineering problems.
    Computer Methods in Applied Mechanics and Engineering, 2024, 432: 117411.
    """
    def __init__(self, option) -> None:
        """
        Initializes the ExponentialTrigonometricOptAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Exponential Trigonometric Optimization Algorithm.
        """
        # Initialize fitness values and find the best solution in the initial population
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        gbest_second = bm.zeros((self.dim,))

        # Initialize parameters
        CEi = 0
        CEi_temp = 0
        UB = self.ub
        LB = self.lb
        T = bm.floor(bm.array(1 + self.MaxIT / 1.55))  # Eq.(2)
        CE = bm.floor(bm.array(1.2 + self.MaxIT / 2.25))  # Eq.(7)

        # Iterate through the maximum number of iterations
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            # Calculate dynamic parameters for exploration and exploitation
            d1 = 0.1 * bm.exp(bm.array(-0.01 * (it + 1))) * bm.cos(bm.array(0.5 * self.MaxIT * (1 - it / self.MaxIT)))  # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(-0.01 * (it + 1))) * bm.cos(bm.array(0.5 * self.MaxIT * (1 - it / self.MaxIT)))  # Eq.(11)
            CM = (bm.sqrt(bm.array((it + 1) / self.MaxIT)) ** bm.tan(d1 / (d2 + 1e-8))) * bm.random.rand(self.N, 1) * 0.01  # Eq.(18)

            # Update bounds dynamically based on the current best solution
            if it == CEi:
                j = bm.random.randint(0, self.dim, (1,))
                r1 = bm.random.rand(1, 1)
                r2 = bm.random.rand(1, 1)
                UB = self.gbest[j] + (1 - it / self.MaxIT) * bm.abs(r2 * self.gbest[j] - gbest_second[j]) * r1  # Eq.(3)
                LB = self.gbest[j] - (1 - it / self.MaxIT) * bm.abs(r2 * self.gbest[j] - gbest_second[j]) * r1  # Eq.(4)
                UB = UB + (self.ub - UB) * (UB > self.ub)
                LB = LB + (self.lb - LB) * (LB < self.lb)
                CEi_temp = CEi
                CEi = 0

            q1 = bm.random.rand(self.N, 1)

            # Calculate alpha values for exploration and exploitation
            d1 = 0.1 * bm.exp(bm.array(0.01 * it)) * bm.cos(0.5 * self.MaxIT * q1)  # Eq.(10)
            d2 = -0.1 * bm.exp(bm.array(0.01 * it)) * bm.cos(0.5 * self.MaxIT * q1)  # Eq.(11)
            alpha_1 = bm.random.rand(self.N, self.dim) * 3 * (it / self.MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1)  # Eq.(9)
            alpha_2 = bm.random.rand(self.N, self.dim) * bm.exp(bm.tanh(1.5 * (-it / self.MaxIT - 0.75) - bm.random.rand(self.N, self.dim)))  # Eq.(13)
            alpha_3 = bm.random.rand(self.N, self.dim) * 3 * (it / self.MaxIT - 0.85) * bm.exp(bm.abs(d1 / d2) - 1.3)  # Eq.(15)

            # First phase: Exploration
            if it < T:
                self.x = (
                    (CM > 1) *
                    (self.gbest + bm.random.rand(self.N, self.dim) * alpha_1 *
                    bm.abs(self.gbest - self.x) * (1 - 2 * (q1 > 0.5))) +  # Eq.(8)
                    (CM <= 1) *
                    (self.gbest + bm.random.rand(self.N, self.dim) * alpha_3 *
                     bm.abs(bm.random.rand(self.N, self.dim) * self.gbest - self.x) * (1 - 2 * (q1 > 0.5))))  # Eq.(14)
            # Second phase: Exploitation
            else:
                self.x = (
                    (CM > 1) *
                    (self.x + bm.exp(bm.tan(bm.abs(d1 / (d2 + 1e-8)) * bm.abs(bm.random.rand(self.N, self.dim) * alpha_2 * self.gbest - self.x)))) +  # Eq.(16)
                    (CM <= 1) *
                    (self.x + 3 * (bm.abs(bm.random.rand(self.N, self.dim) * alpha_2 * self.gbest - self.x)) * (1 - 2 * (q1 > 0.5))))  # Eq.(12)

            # Update bounds and handle boundary conditions
            CEi = CEi_temp
            self.x = self.x + (LB - self.x) * (self.x < LB) + (UB - self.x) * (self.x > UB)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)

            # Update the second-best solution and adjust CEi
            if it == CE:
                CEi = CE + 1
                CE = CE + bm.floor(2 - it * 2 / (self.MaxIT - CE * 4.6) / 1)  # Eq.(1)
                second_index = bm.argsort(fit)[1]
                gbest_second = self.x[second_index]

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f
