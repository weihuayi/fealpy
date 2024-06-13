import taichi as ti
from typing import Union, Tuple

@ti.data_oriented
class CSRMatrix:
    """
    A data class representing a Compressed Sparse Row (CSR) matrix using Taichi.

    Attributes
    ----------
    data : Union[Template, None]
        The non-zero values of the CSR matrix.
    indices : Template
        The column indices of the corresponding elements in `data`.
    indptr : Template
        The index pointers to the start of each row in `data`.
    shape : tuple of int
        The shape of the matrix (rows, columns).
    dtype : data-type
        The data type of the matrix elements.
    """
    def __init__(self, arg1, shape: Tuple[int, int], dtype=None, copy=False):
        """
        Initializes the CSRMatrix with given data, indices, and indptr.

        Parameters
        ----------
        arg1 : 
            A tuple containing the data, indices, and indptr of the CSR matrix.
        shape : tuple of int
            The shape of the matrix (rows, columns).
        dtype : data-type, optional
            The desired data-type for the matrix (default is ti.f64).
        copy : bool, optional
            If True, the data is copied (default is False).

        Raises
        ------
        ValueError
            If `arg1` does not contain exactly three elements.
        """
        if len(arg1) == 3:
            self.data, self.indices, self.indptr = arg1
            self.shape = shape

            if self.data is not None:
                self.dtype = self.data.dtype
            else:
                self.dtype = None

            if copy:
                if self.data is not None:
                    self.data = ti.field(self.data.dtype, shape=self.data.shape)
                    self.data.copy_from(arg1[0])
                self.indices = ti.field(self.indices.dtype, shape=self.indices.shape)
                self.indptr = ti.field(self.indptr.dtype, shape=self.indptr.shape)
                self.indices.copy_from(arg1[1])
                self.indptr.copy_from(arg1[2])
        else:
            raise ValueError(f"Now, we just support arg1 == (data, indices, indptr)!")

    def matvec(self, vec):
        """
        Performs matrix-vector multiplication.

        Parameters
        ----------
        vec : Template
            The input vector to be multiplied.

        Returns
        -------
        result : Template
            The result of the matrix-vector multiplication.
        """
        assert vec.shape[0] == self.shape[1], "Dimension mismatch for matrix-vector multiplication."
        assert self.data is not None, "There is no data in this CSRMatrix instance"

        result = ti.field(self.dtype, shape=(self.shape[0],))
        result.fill(0.0)

        @ti.kernel
        def multiply():
            for i in range(self.shape[0]):
                for j in range(self.indptr[i], self.indptr[i+1]):
                    result[i] += self.data[j] * vec[self.indices[j]]
        multiply()
        return result

    def matmul(self, other):
        """
        Performs matrix-matrix multiplication.

        Parameters
        ----------
        other : CSRMatrix
            The input matrix to be multiplied.

        Returns
        -------
        result : CSRMatrix
            The result of the matrix-matrix multiplication.
        """
        assert self.shape[1] == other.shape[0], "Dimension mismatch for matrix-matrix multiplication."
        pass

