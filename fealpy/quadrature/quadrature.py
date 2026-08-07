
from ..backend import bm, dtype, device, Tensor

type BCS = Tensor | tuple[Tensor, ...]


class Quadrature():
    r"""Base class for quadrature generators."""
    def __init__(
        self,
        index: int,
        *,
        dtype: dtype | None = None,
        device: device | None = None
    ) -> None:
        self.dtype = dtype if dtype else bm.float64
        self.device = device
        self.quadpts, self.weights = self.make(index)

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def __getitem__(self, i: int) -> tuple[BCS, Tensor]:
        return self.get_quadrature_point_and_weight(i)

    def make(self, index: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def number_of_quadrature_points(self) -> int:
        return self.weights.shape[0]

    def get_quadrature_points_and_weights(self) -> tuple[BCS, Tensor]:
        """Get all quadrature points and weights in the formula.

        Returns:
            (Tensor, Tensor): Quadrature points and weights.
        """
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i: int) -> tuple[BCS, Tensor]:
        """Get the i-th quadrature point and weight.

        Parameters:
            i (int): _description_

        Returns:
            (Tensor, Tensor): A quadrature point and weight.
        """
        return self.quadpts[i, :], self.weights[i]
