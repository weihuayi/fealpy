import ipdb
import numpy as np 
import pytest
from fealpy.backend import backend_manager as bm
from backend_data import *

class TestBackendInterfaces:
    ######## Constants
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_e(self, backend):
        '''
        Euler’s number, base of natural logarithms, Napier’s constant.
        https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_euler_gamma(self, backend):
        '''
        Euler's constant, Euler–Mascheroni constant
        
        https://en.wikipedia.org/wiki/Euler-Mascheroni_constant
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_inf(self, backend):
        '''
        IEEE 754 floating point representation of (positive) infinity.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nan(self, backend):
        '''
        IEEE 754 floating point representation of Not a Number (NaN).
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_newaxis(self, backend):
        '''
        A convenient alias for None, useful for indexing arrays.
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pi(self, backend):
        '''
        Pi, The ratio of a circle's circumference to its diameter
        https://en.wikipedia.org/wiki/Pi
        '''
        pass
    
    ######## Array creation routines
    #From shape or value
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_empty(self, backend):
        '''
        Return a new array of given shape and type, without initializing entries.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_empty_like(self, backend):
        '''
        Return a new array with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eye(self, backend):
        '''
        Return a 2-D array with ones on the diagonal and zeros elsewhere.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_identity(self, backend):
        '''
        Return the identity array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ones(self, backend):
        '''
        Return a new array of given shape and type, filled with ones.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ones_like(self, backend):
        '''
        Return an array of ones with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_zeros(self, backend):
        '''
        Return a new array of given shape and type, filled with zeros.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_zeros_like(self, backend):
        '''
        Return an array of zeros with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_full(self, backend):
        '''
        Return a new array of given shape and type, filled with fill_value.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_full_like(self, backend):
        '''
        Return a full array with the same shape and type as a given array.
        '''
        pass

    #From existing data
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array(self, backend):
        '''
        Create an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tensor(self,  backend):
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray(self, backend):
        '''
        Convert the input to an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asanyarray(self, backend):
        '''
        Convert the input to an ndarray, but pass ndarray subclasses through.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ascontiguousarray(self, backend):
        '''
        Return a contiguous array (ndim >= 1) in memory (C order).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asmatrix(self, backend):
        '''
        Interpret the input as a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_astype(self, backend):
        '''
        Copies an array to a specified data type.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_copy(self, backend):
        '''
        Return an array copy of the given object.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_frombuffer(self, backend):
        '''	
        Interpret a buffer as a 1-dimensional array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_from_dlpack(self, backend):
        '''
        Create a NumPy array from an object implementing the __dlpack__ protocol.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromfile(self, backend):
        '''
        Construct an array from data in a text or binary file.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromfunction(self, backend):
        '''
        Construct an array by executing a function over each coordinate.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromiter(self, backend):
        '''	
        Create a new 1-dimensional array from an iterable object.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromstring(self, backend):
        '''
        A new 1-D array initialized from text data in a string.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_loadtxt(self, backend):
        '''
        Load data from a text file.
        '''
        pass

    #Creating record arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_array(self, backend):
        '''
        Construct a record array from a wide-variety of objects.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromarrays(self, backend):
        '''
        Create a record array from a (flat) list of arrays
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromrecords(self, backend):
        '''
        Create a recarray from a list of records in text form.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromstring(self, backend):
        '''
        Create a record array from binary data
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromfile(self, backend):
        '''
        Create an array from binary file data
        '''
        pass
    
    #Creating character arrays 
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_char_array(self, backend):
        '''
        Create a chararray.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_char_asarray(self, backend):
        '''
        Convert the input to a chararray, copying the data only if necessary.
        '''
        pass

    #Numerical ranges
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arange(self, backend):
        '''
        Return evenly spaced values within a given interval.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_linspace(self, backend):
        '''
        Return evenly spaced numbers over a specified interval.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logspace(self, backend):
        '''
        Return numbers spaced evenly on a log scale.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_geomspace(self, backend):
        '''
        Return numbers spaced evenly on a log scale (a geometric progression).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_meshgrid(self, backend):
        '''	
        Return a tuple of coordinate matrices from coordinate vectors.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_mgrid(self, backend):
        '''
        An instance which returns a dense multi-dimensional "meshgrid".
        '''
        pass

    #Building matrices
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diag(self, backend):
        '''
        Extract a diagonal or construct a diagonal array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diagflat(self, backend):
        '''	
        Create a two-dimensional array with the flattened input as a diagonal.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tri(self, backend):
        '''
        An array with ones at and below the given diagonal and zeros elsewhere.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tril(self, backend):
        '''
        Lower triangle of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_triu(self, backend):
        '''
        Upper triangle of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vander(self, backend):
        '''
        Generate a Vandermonde matrix.
        '''
        pass
    
    #The matrix class
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_bmat(self, backend):
        '''
        Build a matrix object from a string, nested sequence, or array.
        '''
        pass
    
    ######## Array manipulation routines
    # Basic operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_copyto(self, backend):
        '''
        Copies values from one array to another, broadcasting as necessary.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndim(self, backend):
        '''
        Return the number of dimensions of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_shape(self, backend):
        '''
        Return the shape of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_size(self, backend):
        '''
        Return the number of elements along a given axis.
        '''
        pass

    # Changing array shape
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_reshape(self, backend):
        '''
        Gives a new shape to an array without changing its data.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ravel(self, backend):
        '''
        Return a contiguous flattened array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_flat(self, backend):
        '''
        A 1-D iterator over the array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_flatten(self, backend):
        '''
        Return a copy of the array collapsed into one dimension.
        '''
        pass

    # Transpose-like operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_moveaxis(self, backend):
        '''
        Move axes of an array to new positions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rollaxis(self, backend):
        '''
        Roll the specified axis backwards, until it lies in a given position.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_swapaxes(self, backend):
        '''
        Interchange two axes of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_T(self, backend):
        '''
        View of the transposed array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_transpose(self, backend):
        '''
        Returns an array with axes transposed.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_permute_dims(self, backend):
        '''
        Returns an array with axes transposed.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_transpose(self, backend):
        '''
        Transposes a matrix (or a stack of matrices) x.
        '''
        pass

    # Changing number of dimensions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_1d(self, backend):
        '''
        Convert inputs to arrays with at least one dimension.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_2d(self, backend):
        '''
        View inputs as arrays with at least two dimensions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_3d(self, backend):
        '''
        View inputs as arrays with at least three dimensions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast(self, backend):
        '''
        Produce an object that mimics broadcasting.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast_to(self, backend):
        '''
        Broadcast an array to a new shape.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast_arrays(self, backend):
        '''
        Broadcast any number of arrays against each other.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_expand_dims(self, backend):
        '''
        Expand the shape of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_squeeze(self, backend):
        '''
        Remove axes of length one from a.
        '''
        pass

    # Changing kind of array
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray(self, backend):
        '''
        Convert the input to an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asanyarray(self, backend):
        '''
        Convert the input to an ndarray, but pass ndarray subclasses through.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asmatrix(self, backend):
        '''
        Interpret the input as a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asfortranarray(self, backend):
        '''
        Return an array (ndim >= 1) laid out in Fortran order in memory.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ascontiguousarray(self, backend):
        '''
        Return a contiguous array (ndim >= 1) in memory (C order).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray_chkfinite(self, backend):
        '''
        Convert the input to an array, checking for NaNs or Infs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_require(self, backend):
        '''
        Return an ndarray of the provided type that satisfies requirements.
        '''
        pass

    # Joining arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_concatenate(self, backend):
        '''
        Join a sequence of arrays along an existing axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_concat(self, backend):
        '''
        Join a sequence of arrays along an existing axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_stack(self, backend):
        '''
        Join a sequence of arrays along a new axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_block(self, backend):
        '''
        Assemble an nd-array from nested lists of blocks.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vstack(self, backend):
        '''
        Stack arrays in sequence vertically (row wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_hstack(self, backend):
        '''
        Stack arrays in sequence horizontally (column wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_dstack(self, backend):
        '''
        Stack arrays in sequence depth wise (along third axis).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_column_stack(self, backend):
        '''
        Stack 1-D arrays as columns into a 2-D array.
        '''
        pass

    # Splitting arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_split(self, backend):
        '''
        Split an array into multiple sub-arrays as views into ary.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array_split(self, backend):
        '''
        Split an array into multiple sub-arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_dsplit(self, backend):
        '''
        Split array into multiple sub-arrays along the 3rd axis (depth).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_hsplit(self, backend):
        '''
        Split an array into multiple sub-arrays horizontally (column-wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vsplit(self, backend):
        '''
        Split an array into multiple sub-arrays vertically (row-wise).
        '''
        pass

    # Tiling arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tile(self, backend):
        '''
        Construct an array by repeating A the number of times given by reps.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_repeat(self, backend):
        '''
        Repeat each element of an array after themselves.
        '''
        pass

    # Adding and removing elements
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_delete(self, backend):
        '''
        Return a new array with sub-arrays along an axis deleted.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_insert(self, backend):
        '''
        Insert values along the given axis before the given indices.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_append(self, backend):
        '''
        Append values to the end of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_resize(self, backend):
        '''
        Return a new array with the specified shape.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_trim_zeros(self, backend):
        '''
        Trim the leading and/or trailing zeros from a 1-D array or sequence.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_unique(self, backend):
        '''
        Find the unique elements of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pad(self, backend):
        '''
        Pad an array.
        '''
        pass

    # Rearranging elements
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flip(self, backend):
        '''
        Reverse the order of elements in an array along the given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fliplr(self, backend):
        '''
        Reverse the order of elements along axis 1 (left/right).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flipud(self, backend):
        '''
        Reverse the order of elements along axis 0 (up/down).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_reshape(self, backend):
        '''
        Gives a new shape to an array without changing its data.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_roll(self, backend):
        '''
        Roll array elements along a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rot90(self, backend):
        '''
        Rotate an array by 90 degrees in the plane specified by axes.
        '''
        pass
    
    ######## Functional programming
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_apply_along_axis(self, backend):
        '''
        Apply a function to 1-D slices along the given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_apply_over_axes(self, backend):
        '''
        Apply a function repeatedly over multiple axes.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vectorize(self, backend):
        '''
        Returns an object that acts like pyfunc, but takes arrays as input.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_frompyfunc(self, backend):
        '''
        Takes an arbitrary Python function and returns a NumPy ufunc.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_piecewise(self, backend):
        '''
        Evaluate a piecewise-defined function.
        '''
        pass
    
    ######## Linear algebra
    # Matrix and vector products
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_dot(self, backend):
        '''
        Dot product of two arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_multi_dot(self, backend):
        '''
        Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vdot(self, backend):
        '''
        Return the dot product of two vectors.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vecdot(self, backend):
        '''
        Vector dot product of two arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_inner(self, backend):
        '''
        Inner product of two arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_outer(self, backend):
        '''
        Compute the outer product of two vectors.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matmul(self, backend):
        '''
        Matrix product of two arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tensordot(self, backend):
        '''
        Compute tensor dot product along specified axes.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_einsum(self, backend):
        '''
        Evaluates the Einstein summation convention on the operands.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_einsum_path(self, backend):
        '''
        Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_power(self, backend):
        '''
        Raise a square matrix to the (integer) power n.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_kron(self, backend):
        '''
        Kronecker product of two arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cross(self, backend):
        '''
        Returns the cross product of 3-element vectors.
        '''
        pass

    # Decompositions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cholesky(self, backend):
        '''
        Cholesky decomposition.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_qr(self, backend):
        '''
        Compute the qr factorization of a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_svd(self, backend):
        '''
        Singular Value Decomposition.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eig(self, backend):
        '''
        Compute the eigenvalues and right eigenvectors of a square array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eigh(self, backend):
        '''
        Return the eigenvalues and eigenvectors of a complex Hermitian or a real symmetric matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eigvals(self, backend):
        '''
        Compute the eigenvalues of a general matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eigvalsh(self, backend):
        '''
        Compute the eigenvalues of a complex Hermitian or real symmetric matrix.
        '''
        pass

    # Norms and other numbers
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_norm(self, backend):
        '''
        Matrix or vector norm.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_norm(self, backend):
        '''
        Computes the matrix norm of a matrix (or a stack of matrices) x.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vector_norm(self, backend):
        '''
        Computes the vector norm of a vector (or batch of vectors) x.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cond(self, backend):
        '''
        Compute the condition number of a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_det(self, backend):
        '''
        Compute the determinant of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_rank(self, backend):
        '''
        Return matrix rank of array using SVD method.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_slogdet(self, backend):
        '''
        Compute the sign and (natural) logarithm of the determinant of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_trace(self, backend):
        '''
        Return the sum along diagonals of the array.
        '''
        pass

    # Solving equations and inverting matrices
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_solve(self, backend):
        '''
        Solve a linear matrix equation, or system of linear scalar equations.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tensorsolve(self, backend):
        '''
        Solve the tensor equation a x = b for x.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_lstsq(self, backend):
        '''
        Return the least-squares solution to a linear matrix equation.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_inv(self, backend):
        '''
        Compute the inverse of a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pinv(self, backend):
        '''
        Compute the (Moore-Penrose) pseudo-inverse of a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tensorinv(self, backend):
        '''
        Compute the 'inverse' of an N-dimensional array.
        '''
        pass

    # Other matrix operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diagonal(self, backend):
        '''
        Return specified diagonals.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_transpose(self, backend):
        '''
        Transposes a matrix (or a stack of matrices) x.
        '''
        pass

    # Exceptions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_LinAlgError(self, backend):
        '''
        Generic Python-exception-derived object raised by linalg functions.
        '''
        pass
    
    ####### Logic funtions
    ## Truth value testing
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_all(self, backend):
        '''
        Test whether all array elements along a given axis evaluate to True.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_any(self, backend):
        '''
        Test whether any array element along a given axis evaluates to True.
        '''
        pass
    
    ## Array contents
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isfinite(self, backend):
        '''
        Test element-wise for finiteness (not infinity and not Not a Number).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isinf(self, backend):
        '''
        Test element-wise for positive or negative infinity.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isnan(self, backend):
        '''
        Test element-wise for NaN and return result as a boolean array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isnat(self, backend):
        '''
        Test element-wise for NaT (not a time) and return result as a boolean array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isneginf(self, backend):
        '''
        Test element-wise for negative infinity, return result as bool array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isposinf(self, backend):
        '''
        Test element-wise for positive infinity, return result as bool array.
        '''
        pass
    
    ## Arrat type testing
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_iscomplex(self, backend):
        '''
        Returns a bool array, where True if input element is complex.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_iscomplexobj(self, backend):
        '''
        Check for a complex type or an array of complex numbers.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isfortran(self, backend):
        '''
        Check if the array is Fortran contiguous but not C contiguous.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isreal(self, backend):
        '''
        Returns a bool array, where True if input element is real.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isrealobj(self, backend):
        '''
        Return True if x is a not complex type or an array of complex numbers.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isscalar(self, backend):
        '''
        Returns True if the type of element is a scalar type.
        '''
        pass
    
    ## Logical operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logical_and(self, backend):
        '''
        Compute the truth value of x1 AND x2 element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logical_or(self, backend):
        '''
        Compute the truth value of x1 OR x2 element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logical_not(self, backend):
        '''
        Compute the truth value of NOT x element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logical_xor(self, backend):
        '''
        Compute the truth value of x1 XOR x2, element-wise.
        '''
        pass
    
    ## Comparsion
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_allclose(self, backend):
        '''
        Returns True if two arrays are element-wise equal within a tolerance.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_isclose(self, backend):
        '''
        Returns a boolean array where two arrays are element-wise equal within a tolerance.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array_equal(self, backend):
        '''
        True if two arrays have the same shape and elements, False otherwise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array_equiv(self, backend):
        '''
        Returns True if input arrays are shape consistent and all elements equal.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_greater(self, backend):
        '''
        Return the truth value of (x1 > x2) element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_greater_equal(self, backend):
        '''
        Return the truth value of (x1 >= x2) element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_less(self, backend):
        '''
        Return the truth value of (x1 < x2) element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_less_equal(self, backend):
        '''
        Return the truth value of (x1 <= x2) element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_equal(self, backend):
        '''
        Return (x1 == x2) element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_not_equal(self, backend):
        '''
        Return (x1 != x2) element-wise.
        '''
        pass

    ######## Inedxing routines
    # Generating index arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_c_(self, backend):
        '''
        Translates slice objects to concatenation along the second axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_r_(self, backend):
        '''
        Translates slice objects to concatenation along the first axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_s_(self, backend):
        '''
        A nicer way to build up index tuples for arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nonzero(self, backend):
        '''
        Return the indices of the elements that are non-zero.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_where(self, backend):
        '''
        Return elements chosen from x or y depending on condition.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_indices(self, backend):
        '''
        Return an array representing the indices of a grid.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ix_(self, backend):
        '''
        Construct an open mesh from multiple sequences.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ogrid(self, backend):
        '''
        An instance which returns an open multi-dimensional "meshgrid".
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ravel_multi_index(self, backend):
        '''
        Converts a tuple of index arrays into an array of flat indices, applying boundary modes to the multi-index.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_unravel_index(self, backend):
        '''
        Converts a flat index or array of flat indices into a tuple of coordinate arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diag_indices(self, backend):
        '''
        Return the indices to access the main diagonal of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diag_indices_from(self, backend):
        '''
        Return the indices to access the main diagonal of an n-dimensional array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_mask_indices(self, backend):
        '''
        Return the indices to access (n, n) arrays, given a masking function.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tril_indices(self, backend):
        '''
        Return the indices for the lower-triangle of an (n, m) array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tril_indices_from(self, backend):
        '''
        Return the indices for the lower-triangle of arr.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_triu_indices(self, backend):
        '''
        Return the indices for the upper-triangle of an (n, m) array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_triu_indices_from(self, backend):
        '''
        Return the indices for the upper-triangle of arr.
        '''
        pass

    # Indexing-like operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_take(self, backend):
        '''
        Take elements from an array along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_take_along_axis(self, backend):
        '''
        Take values from the input array by matching 1d index and data slices.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_choose(self, backend):
        '''
        Construct an array from an index array and a list of arrays to choose from.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_compress(self, backend):
        '''
        Return selected slices of an array along given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_select(self, backend):
        '''
        Return an array drawn from elements in choicelist, depending on conditions.
        '''
        pass

    # Inserting data into arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_place(self, backend):
        '''
        Change elements of an array based on conditional and input values.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_put(self, backend):
        '''
        Replaces specified elements of an array with given values.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_put_along_axis(self, backend):
        '''
        Put values into the destination array by matching 1d index and data slices.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_putmask(self, backend):
        '''
        Changes elements of an array based on conditional and input values.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fill_diagonal(self, backend):
        '''
        Fill the main diagonal of the given array of any dimensionality.
        '''
        pass

    # Iterating over arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nditer(self, backend):
        '''
        Efficient multi-dimensional iterator object to iterate over arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndenumerate(self, backend):
        '''
        Multidimensional index iterator.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndindex(self, backend):
        '''
        An N-dimensional iterator object to index arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nested_iters(self, backend):
        '''
        Create nditers for use in nested loops.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flatiter(self, backend):
        '''
        Flat iterator object to iterate over arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_iterable(self, backend):
        '''
        Check whether or not an object can be iterated over.
        '''
        pass
    
    ######## Mathematical functions
    # Trigonometric functions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sin(self, backend):
        '''
        Trigonometric sine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cos(self, backend):
        '''
        Cosine element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tan(self, backend):
        '''
        Compute tangent element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arcsin(self, backend):
        '''
        Inverse sine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asin(self, backend):
        '''
        Inverse sine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arccos(self, backend):
        '''
        Trigonometric inverse cosine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_acos(self, backend):
        '''
        Trigonometric inverse cosine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arctan(self, backend):
        '''
        Trigonometric inverse tangent, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atan(self, backend):
        '''
        Trigonometric inverse tangent, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_hypot(self, backend):
        '''
        Given the "legs" of a right triangle, return its hypotenuse.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arctan2(self, backend):
        '''
        Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atan2(self, backend):
        '''
        Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_degrees(self, backend):
        '''
        Convert angles from radians to degrees.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_radians(self, backend):
        '''
        Convert angles from degrees to radians.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_unwrap(self, backend):
        '''
        Unwrap by taking the complement of large deltas with respect to the period.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_deg2rad(self, backend):
        '''
        Convert angles from degrees to radians.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rad2deg(self, backend):
        '''
        Convert angles from radians to degrees.
        '''
        pass

    # Hyperbolic functions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sinh(self, backend):
        '''
        Hyperbolic sine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cosh(self, backend):
        '''
        Hyperbolic cosine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tanh(self, backend):
        '''
        Compute hyperbolic tangent element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arcsinh(self, backend):
        '''
        Inverse hyperbolic sine element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asinh(self, backend):
        '''
        Inverse hyperbolic sine element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arccosh(self, backend):
        '''
        Inverse hyperbolic cosine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_acosh(self, backend):
        '''
        Inverse hyperbolic cosine, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arctanh(self, backend):
        '''
        Inverse hyperbolic tangent element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atanh(self, backend):
        '''
        Inverse hyperbolic tangent element-wise.
        '''
        pass

    # Rounding
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_round(self, backend):
        '''
        Evenly round to the given number of decimals.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_around(self, backend):
        '''
        Round an array to the given number of decimals.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rint(self, backend):
        '''
        Round elements of the array to the nearest integer.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fix(self, backend):
        '''
        Round to nearest integer towards zero.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_floor(self, backend):
        '''
        Return the floor of the input, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ceil(self, backend):
        '''
        Return the ceiling of the input, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_trunc(self, backend):
        '''
        Return the truncated value of the input, element-wise.
        '''
        pass

    # Sums, products, differences
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_prod(self, backend):
        '''
        Return the product of array elements over a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sum(self, backend):
        '''
        Sum of array elements over a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nanprod(self, backend):
        '''
        Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nansum(self, backend):
        '''
        Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cumprod(self, backend):
        '''
        Return the cumulative product of elements along a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cumsum(self, backend):
        '''
        Return the cumulative sum of the elements along a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nancumprod(self, backend):
        '''
        Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nancumsum(self, backend):
        '''
        Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diff(self, backend):
        '''
        Calculate the n-th discrete difference along the given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ediff1d(self, backend):
        '''
        The differences between consecutive elements of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_gradient(self, backend):
        '''
        Return the gradient of an N-dimensional array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cross(self, backend):
        '''
        Return the cross product of two (arrays of) vectors.
        '''
        pass

    # Exponents and logarithms
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_exp(self, backend):
        '''
        Calculate the exponential of all elements in the input array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_expm1(self, backend):
        '''
        Calculate exp(x) - 1 for all elements in the array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_exp2(self, backend):
        '''
        Calculate 2**p for all p in the input array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_log(self, backend):
        '''
        Natural logarithm, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_log10(self, backend):
        '''
        Return the base 10 logarithm of the input array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_log2(self, backend):
        '''
        Base-2 logarithm of x.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_log1p(self, backend):
        '''
        Return the natural logarithm of one plus the input array, element-wise.
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logaddexp(self, backend):
        '''
        Logarithm of the sum of exponentiations of the inputs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logaddexp2(self, backend):
        '''
        Logarithm of the sum of exponentiations of the inputs in base-2.
        '''
        pass
    
    #Other special functions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_i0(self, backend):
        '''
        Modified Bessel function of the first kind, order 0.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sinc(self, backend):
        '''
        Return the normalized sinc function.
        '''
        pass
    
    #Floating point routines 
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_signbit(self, backend):
        '''
        Returns element-wise True where signbit is set (less than zero).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_copysign(self, backend):
        '''
        Change the sign of x1 to that of x2, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_frexp(self, backend):
        '''
        Decompose the elements of x into mantissa and twos exponent.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ldexp(self, backend):
        '''
        Returns x1 * 2**x2, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nextafter(self, backend):
        '''
        Return the next floating-point value after x1 towards x2, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_spacing(self, backend):
        '''
        Return the distance between x and the nearest adjacent number.
        '''
        pass
    
    #Rational routines 
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_lcm(self, backend):
        '''
        Returns the lowest common multiple of |x1| and |x2|.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_gcd(self, backend):
        '''
        Returns the greatest common divisor of |x1| and |x2|.
        '''
        pass

    #Arithmetic operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_add(self, backend):
        '''
        Add arguments element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_reciprocal(self, backend):
        '''
        Return the reciprocal of the argument, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_positive(self, backend):
        '''
        Numerical positive, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_negative(self, backend):
        '''
        Numerical negative, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_multiply(self, backend):
        '''
        Multiply arguments element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_divide(self, backend):
        '''
        Divide arguments element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_power(self, backend):
        '''
        First array elements raised to powers from second array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pow(self, backend):
        '''
        First array elements raised to powers from second array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_subtract(self, backend):
        '''
        Subtract arguments, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_true_divide(self, backend):
        '''
        Divide arguments element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_floor_divide(self, backend):
        '''
        Return the largest integer smaller or equal to the division of the inputs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_float_power(self, backend):
        '''
        First array elements raised to powers from second array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fmod(self, backend):
        '''
        Returns the element-wise remainder of division.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_mod(self, backend):
        '''
        Returns the element-wise remainder of division.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_modf(self, backend):
        '''
        Return the fractional and integral parts of an array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_remainder(self, backend):
        '''
        Returns the element-wise remainder of division.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_divmod(self, backend):
        '''
        Return element-wise quotient and remainder simultaneously.
        '''
        pass
    
    #Handling complex numbers
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_angle(self, backend):
        '''
        Return the angle of the complex argument.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_real(self, backend):
        '''
        Return the real part of the complex argument.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_imag(self, backend):
        '''
        Return the imaginary part of the complex argument.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_conj(self, backend):
        '''
        Return the complex conjugate, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_conjugate(self, backend):
        '''
        Return the complex conjugate, element-wise.
        '''
        pass
    
    #Extrema finding
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_maximum(self, backend):
        '''
        Element-wise maximum of array elements.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_max(self, backend):
        '''
        Return the maximum of an array or maximum along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_amax(self, backend):
        '''
        Return the maximum of an array or maximum along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fmax(self, backend):
        '''
        Element-wise maximum of array elements.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nanmax(self, backend):
        '''
        Return the maximum of an array or maximum along an axis, ignoring any NaNs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_minimum(self, backend):
        '''
        Element-wise minimum of array elements.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_min(self, backend):
        '''
        Return the minimum of an array or minimum along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_amin(self, backend):
        '''
        Return the minimum of an array or minimum along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fmin(self, backend):
        '''
        Element-wise minimum of array elements.
        '''
        pass

    #Miscellaneous
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nanmin(self, backend):
        '''
        Return minimum of an array or minimum along an axis, ignoring any NaNs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_convolve(self, backend):
        '''
        Returns the discrete, linear convolution of two one-dimensional sequences.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_clip(self, backend):
        '''
        Clip (limit) the values in an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sqrt(self, backend):
        '''
        Return the non-negative square-root of an array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_cbrt(self, backend):
        '''
        Return the cube-root of an array, element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_square(self, backend):
        '''
        Return the element-wise square of the input.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_absolute(self, backend):
        '''
        Calculate the absolute value element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fabs(self, backend):
        '''
        Compute the absolute values element-wise.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sign(self, backend):
        '''
        Returns an element-wise indication of the sign of a number.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_heaviside(self, backend):
        '''
        Compute the Heaviside step function.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nan_to_num(self, backend):
        '''
        Replace NaN with zero and infinity with large finite numbers (default behaviour) or with the numbers defined by the user using the nan, posinf and/or neginf keywords.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_real_if_close(self, backend):
        '''
        If input is complex with all imaginary parts close to zero, return real parts.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_interp(self, backend):
        '''
        One-dimensional linear interpolation for monotonically increasing sample points.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_bitwise_count(self, backend):
        '''
        Computes the number of 1-bits in the absolute value of x.
        '''
        pass
    
    ######## Random

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_random_rand(self, backend):
        '''
        Random values in a given shape.
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_random_choice(self, backend):
        '''
        Generates a random sample from a given 1-D array
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_random_randn(self, backend):
        '''
        Return a sample (or samples) from the "standard normal" distribution.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_random_randint(self, backend):
        '''
        Return random integers from low (inclusive) to high (exclusive).
        '''
        pass
    
    ####### Sorting, searching, and counting
    ## Sorting
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sort(self, backend):
        '''
        Return a sorted copy of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_lexsort(self, backend):
        '''
        Perform an indirect stable sort using a sequence of keys.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_argsort(self, backend):
        '''
        Returns the indices that would sort an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_sort(self, backend):
        '''
        Sort an array in-place.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sort_complex(self, backend):
        '''
        Sort a complex array using the real part first, then the imaginary part.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_partition(self, backend):
        '''
        Return a partitioned copy of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_argpartition(self, backend):
        '''
        Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.
        '''
        pass
    
    ## Searching
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_argmax(self, backend):
        '''
        Returns the indices of the maximum values along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nanargmax(self, backend):
        '''
        Return the indices of the maximum values in the specified axis ignoring NaNs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_argmin(self, backend):
        '''
        Returns the indices of the minimum values along an axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nanargmin(self, backend):
        '''
        Return the indices of the minimum values in the specified axis ignoring NaNs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_argwhere(self, backend):
        '''
        Find the indices of array elements that are non-zero, grouped by element.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nonzero(self, backend):
        '''
        Return the indices of the elements that are non-zero.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flatnonzero(self, backend):
        '''
        Return indices that are non-zero in the flattened version of a.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_where(self, backend):
        '''
        Return elements chosen from x or y depending on condition.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_searchsorted(self, backend):
        '''
        Find indices where elements should be inserted to maintain order.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_extract(self, backend):
        '''
        Return the elements of an array that satisfy some condition.
        '''
        pass
    
    ## Counting
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_count_nonzero(self, backend):
        '''
        Counts the number of non-zero values in the array a
        '''
        pass



    ######## Fealpy Function
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", unique_data)
    def test_unique(self, backend, data):
        bm.set_backend(backend)
        name = ('result', 'indices', 'inverse', 'counts')
        indata = data["input"]
        result = data["result"]
        test_result = bm.unique(bm.from_numpy(indata), return_index=True, 
                             return_inverse=True, 
                             return_counts=True, axis=0) 
        
        for r, e, s in zip(result, test_result, name):
            np.testing.assert_array_equal(r, bm.to_numpy(e), 
                                          err_msg=f"The {s} of `bm.unique` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", multi_index_data)
    def test_multi_index_matrix(self, backend, data):
        bm.set_backend(backend)
        p = data["p"]
        dim = data["dim"]
        result = data["result"]
        test_result = bm.multi_index_matrix(p, dim)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.multi_index_matrix` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_length(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_length"]
        test_result = bm.edge_length(edge, node)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_length` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_normal(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_normal"]
        test_result = bm.edge_normal(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_normal` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_tangent(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_tangent"]
        test_result = bm.edge_tangent(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                     err_msg=f" `bm.edge_tangent` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_bc_to_points(self, backend, data):
        bm.set_backend(backend)
        bcs = bm.from_numpy(data["bcs"])
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["bc_to_points"]
        test_result = bm.bc_to_points(bcs, node, cell)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                     err_msg=f" `bm.bc_to_points` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_barycenter(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result_cell = data["cell_barycenter"]
        result_edge = data["edge_barycenter"]
        test_result_cell = bm.barycenter(cell, node)
        test_result_edge = bm.barycenter(edge, node)
        np.testing.assert_array_equal(result_cell, bm.to_numpy(test_result_cell), 
                                     err_msg=f" cell of `bm.barycenter` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(result_edge, bm.to_numpy(test_result_edge), 
                                     err_msg=f" edge of `bm.barycenter` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_measure(self, backend, data):
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["simple_measure"]
        test_result = bm.simplex_measure(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=15, 
                                     err_msg=f" `bm.simple_measure` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_shape_function(self, backend, data):
        bm.set_backend(backend)
        bcs = bm.from_numpy(data["bcs"])
        p = data["p"]
        result = data["simple_shape_function"]
        test_result = bm.simplex_shape_function(bcs, p)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.simple_shape_function` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_grad_shape_function(self, backend, data):
        bm.set_backend(backend)
        bcs = bm.from_numpy(data["bcs"])
        p = data["p"]
        result = data["simple_grad_shape_function"]
        test_result = bm.simplex_grad_shape_function(bcs, p)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.simple_grad_shape_function` function is not equal to real result in backend {backend}")
    
    ##TODO:HESS_SHAPE_FUNCTION TEST    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", interval_mesh_data)
    def test_interval_grad_lambda(self, backend, data): 
        bm.set_backend(backend)
        line = bm.from_numpy(data["line"])
        node = bm.from_numpy(data["node"])
        result = data["interval_grad_lambda"]
        test_result = bm.interval_grad_lambda(line, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.interval_grad_lambda` function is not equal to real result in backend {backend}")
    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh3d_data)
    def test_triangle_area_3d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_area_3d"]
        test_result = bm.triangle_area_3d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_area_3d` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_triangle_grad_lambda_2d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_grad_lambda_2d"]
        test_result = bm.triangle_grad_lambda_2d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_grad_lambda_2d` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", triangle_mesh3d_data)
    def test_triangle_grad_lambda_3d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_grad_lambda_3d"]
        test_result = bm.triangle_grad_lambda_3d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_grad_lambda_3d` function is not equal to real result in backend {backend}")
    
    ##TODO:QUADRANGLE_GRAD_LAMBDA_2D

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax', 'cupy', 'paddle'])
    @pytest.mark.parametrize("data", tetrahedron_mesh_data)
    def test_tetrahedron_grad_lambda_3d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        localface = bm.from_numpy(data["localface"])
        result = data["tetrahedron_grad_lambda_3d"]
        test_result = bm.tetrahedron_grad_lambda_3d(cell, node, localface)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.tetrahedron_grad_lambda_3d` function is not equal to real result in backend {backend}")






if __name__ == "__main__":
    pytest.main(['test_backends.py'])
