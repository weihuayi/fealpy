from builtins import bool, float, dict, str

class ConstraintConditions:
    def __init__(self):
        """
        Initialize global constraint conditions.

        This class manages different types of constraints used in topology optimization,
        starting with a default volume fraction constraint.

        Attributes:
            constraints (dict): A dictionary holding the state and values of different constraints.
        """
        self.constraints: dict = {}

    def set_volume_constraint(self, is_on: bool, vf: float):
        """
        Set the volume fraction constraint.

        Args:
            is_on (bool): Indicates whether the volume fraction constraint is enabled.
            vf (float): The target volume fraction value.
        """
        self.constraints['volume'] = {'isOn': is_on, 'vf': vf}

    def get_constraints(self) -> dict:
        """
        Return all the constraints.

        Returns:
            dict: A dictionary containing the current state and values of all constraints.
        """
        return self.constraints

    def __repr__(self) -> str:
        """
        Return a string representation of the global constraints.

        Returns:
            str: A string showing the current state of global constraints.
        """
        return f"ConstraintConditions({self.constraints})"
