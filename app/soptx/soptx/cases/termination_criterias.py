from builtins import bool, float

class TerminationCriteria:
    def __init__(self, max_loop: int, tol_change: float):
        """
        Initialize the termination criteria for the optimization loop.

        Args:
            max_loop (int): The maximum number of iterations (loop count).
            tol_change (float): The tolerance for the change in design variables 
                                to determine convergence.
        """
        self.max_loop = max_loop
        self.tol_change = tol_change

    def should_terminate(self, current_loop: int, current_change: float) -> bool:
        """
        Check whether the optimization loop should terminate.

        Args:
            current_loop (int): The current iteration count.
            current_change (float): The current change in design variables.

        Returns:
            bool: True if the loop should terminate, False otherwise.
        """
        return current_loop >= self.max_loop or current_change <= self.tol_change

    def __repr__(self):
        """
        Return a string representation of the termination criteria.

        Returns:
            str: A string showing the max_loop and tol_change values.
        """
        return (f"TerminationCriteria(max_loop={self.max_loop}, "
                f"tol_change={self.tol_change})")
