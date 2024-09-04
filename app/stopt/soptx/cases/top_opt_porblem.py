from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from builtins import float

class TopOptProblem:
    """
    Base class for topology optimization problems.
    """

    def __init__(self, name: str):
        """
        Initialize the topology optimization problem with a given name.

        Args:
            name (str): The name of the optimization problem.
        """
        self.name = name

    def compute_objective(self, *args, **kwargs) -> float:
        """
        Compute the objective function value. To be implemented by subclasses.

        Returns:
            float: The computed objective function value.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ComplianceMinimization(TopOptProblem):
    """
    Compliance minimization problem for topology optimization.
    """

    def __init__(self):
        """
        Initialize the compliance minimization problem.
        """
        super().__init__(name="Compliance Minimization")

    def compute_objective(self, uhe: TensorLike, KE: TensorLike, E: TensorLike) -> float:
        """
        Compute the compliance objective function value.

        Args:
            uhe (TensorLike): Displacement field values for each element.
            KE (TensorLike): Element stiffness matrix for E = 1.
            E (TensorLike): Young's modulus for each element.

        Returns:
            float: The compliance objective function value.
        """
        ce = self.compute_element_compliance(uhe, KE)
        c = bm.einsum('c, c -> ', E, ce)
        return c

    def compute_element_compliance(self, uhe: TensorLike, KE: TensorLike) -> TensorLike:
        """
        Compute the compliance for each element.

        Args:
            uhe (TensorLike): Displacement field values for each element.
            KE (TensorLike): Element stiffness matrix for E = 1.

        Returns:
            TensorLike: Element-wise compliance values.
        """
        ce = bm.einsum('ci, cik, ck -> c', uhe, KE, uhe)
        return ce
