import taichi as ti
from typing import Union, Tuple


@ti.data_oriented
class CSRMatrix:
    def __init__(self, arg1, shape: Tuple[int, int], dtype=None, itype=None, copy=False):
        if len(arg1) == 3:
            self.data, self.indices, self.indptr = arg1
            self.shape = shape

            if self.data is not None:
                if dtype is not None:
                    self.dtype = dtype
                else:
                    self.dtype = self.data.dtype
            else:
                self.dtype = None 

            if itype is None:
                self.itype = self.indices.dtype
            else:
                self.itype = itype

            if copy:
                if self.data is not None:
                    self.data = ti.field(self.dtype, shape=self.data.shape)
                    self.data.copy_from(arg1[0])

                self.indices = ti.field(self.itype, shape=self.indices.shape)
                self.indptr = ti.field(self.itype, shape=self.indptr.shape)

                self.indices.copy_from(arg1[1])
                self.indptr.copy_from(arg1[2])
        else:
            raise ValueError(f"Now, we just support arg1 == (data, indices, indptr)!")

    @classmethod
    def from_scipy(cls, scipy_csr, dtype=None, itype=None):
        """
        Creates a CSRMatrix instance from a scipy.sparse.csr_matrix.

        Parameters
        ----------
        scipy_csr : scipy.sparse.csr_matrix
            The input scipy CSR matrix.

        Returns
        -------
        CSRMatrix
            The resulting CSRMatrix instance.
        """
        from ..utils import numpy_to_taichi_dtype
        if dtype is None:
            dtype = numpy_to_taichi_dtype(scipy_csr.data.dtype)

        if itype is None:
            itype = numpy_to_taichi_dtype(scipy_csr.indices.dtype)

        data = ti.field(dtype=dtype, shape=scipy_csr.data.shape)
        indices = ti.field(dtype=itype, shape=scipy_csr.indices.shape)
        indptr = ti.field(dtype=itype, shape=scipy_csr.indptr.shape)

        data.from_numpy(scipy_csr.data)
        indices.from_numpy(scipy_csr.indices)
        indptr.from_numpy(scipy_csr.indptr)

        shape = scipy_csr.shape

        return cls((data, indices, indptr), shape=shape, dtype=dtype, itype=itype)

    def __str__(self):
        """
        Generates a string representation of the CSR matrix.

        Returns
        -------
        str
            The string representation of the CSR matrix.
        """
        matrix_str = ""
        for i in range(self.shape[0]):
            for j in range(self.indptr[i], self.indptr[i+1]):
                matrix_str += f"({i}, {self.indices[j]})\t{self.data[j]}\n"
        return matrix_str.strip()

    def __mul__(self, vec):
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

        if self.data is not None:
            result = ti.field(self.dtype, shape=(self.shape[0],))
            result.fill(0)

            @ti.kernel
            def multiply():
                for i in range(self.shape[0]):
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        result[i] += self.data[j] * vec[self.indices[j]]
            multiply()
            return result
        else:
            result = ti.field(vec.dtype, shape=(self.shape[0],))
            result.fill(0)

            @ti.kernel
            def multiply():
                for i in range(self.shape[0]):
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        result[i] += vec[self.indices[j]]
            multiply()
            return result

    __matmul__ = __mul__

    def tofield(self):
        """
        Converts the CSR matrix to a Taichi field representation.

        Returns
        -------
        field : ti.field
            The Taichi field representation of the CSR matrix.
        """
        if self.data is not None:
            result = ti.field(self.dtype, shape=self.shape)

            @ti.kernel
            def fill_field():
                for i in range(self.shape[0]):
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        result[i, self.indices[j]] = self.data[j]
            fill_field()
            return result
        else:
            result = ti.field(ti.i8, shape=self.shape)

            @ti.kernel
            def fill_field():
                for i in range(self.shape[0]):
                    for j in range(self.indptr[i], self.indptr[i+1]):
                        result[i, self.indices[j]] = 1 
            fill_field()
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
