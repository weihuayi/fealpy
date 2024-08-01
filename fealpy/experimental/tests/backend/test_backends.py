import ipdb
import numpy as np 
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.tests.backend.backend_data import *

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
    def test_inf(self,backend):
        '''
        IEEE 754 floating point representation of (positive) infinity.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_nan(self,backend):
        '''
        IEEE 754 floating point representation of Not a Number (NaN).
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_newaxis(self,backend):
        '''
        A convenient alias for None, useful for indexing arrays.
        '''
        pass
    
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pi(self,backend):
        '''
        Pi, The ratio of a circle's circumference to its diameter
        https://en.wikipedia.org/wiki/Pi
        '''
        pass
    
    ######## Array creation routines
    #From shape or value
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_empty(self,backend):
        '''
        Return a new array of given shape and type, without initializing entries.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_empty_like(self,backend):
        '''
        Return a new array with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_eye(self,backend):
        '''
        Return a 2-D array with ones on the diagonal and zeros elsewhere.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_identity(self,backend):
        '''
        Return the identity array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ones(self,backend):
        '''
        Return a new array of given shape and type, filled with ones.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ones_like(self,backend):
        '''
        Return an array of ones with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_zeros(self,backend):
        '''
        Return a new array of given shape and type, filled with zeros.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_zeros_like(self,backend):
        '''
        Return an array of zeros with the same shape and type as a given array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_full(self,backend):
        '''
        Return a new array of given shape and type, filled with fill_value.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_full_like(self,backend):
        '''
        Return a full array with the same shape and type as a given array.
        '''
        pass

    #From existing data
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array(self,backend):
        '''
        Create an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tensor(self, backend):
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray(self,backend):
        '''
        Convert the input to an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asanyarray(self,backend):
        '''
        Convert the input to an ndarray, but pass ndarray subclasses through.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ascontiguousarray(self,backend):
        '''
        Return a contiguous array (ndim >= 1) in memory (C order).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asmatrix(self,backend):
        '''
        Interpret the input as a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_astype(self,backend):
        '''
        Copies an array to a specified data type.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_copy(self,backend):
        '''
        Return an array copy of the given object.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_frombuffer(self,backend):
        '''	
        Interpret a buffer as a 1-dimensional array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_from_dlpack(self,backend):
        '''
        Create a NumPy array from an object implementing the __dlpack__ protocol.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromfile(self,backend):
        '''
        Construct an array from data in a text or binary file.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromfunction(self,backend):
        '''
        Construct an array by executing a function over each coordinate.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromiter(self,backend):
        '''	
        Create a new 1-dimensional array from an iterable object.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fromstring(self,backend):
        '''
        A new 1-D array initialized from text data in a string.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_loadtxt(self,backend):
        '''
        Load data from a text file.
        '''
        pass

    #Creating record arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_array(self,backend):
        '''
        Construct a record array from a wide-variety of objects.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromarrays(self,backend):
        '''
        Create a record array from a (flat) list of arrays
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromrecords(self,backend):
        '''
        Create a recarray from a list of records in text form.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromstring(self,backend):
        '''
        Create a record array from binary data
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rec_fromfile(self,backend):
        '''
        Create an array from binary file data
        '''
        pass
    
    #Creating character arrays 
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_char_array(self,backend):
        '''
        Create a chararray.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_char_asarray(self,backend):
        '''
        Convert the input to a chararray, copying the data only if necessary.
        '''
        pass

    #Numerical ranges
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_arange(self,backend):
        '''
        Return evenly spaced values within a given interval.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_linspace(self,backend):
        '''
        Return evenly spaced numbers over a specified interval.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_logspace(self,backend):
        '''
        Return numbers spaced evenly on a log scale.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_geomspace(self,backend):
        '''
        Return numbers spaced evenly on a log scale (a geometric progression).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_meshgrid(self,backend):
        '''	
        Return a tuple of coordinate matrices from coordinate vectors.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_mgrid(self,backend):
        '''
        An instance which returns a dense multi-dimensional "meshgrid".
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ogrid(self,backend):
        '''
        An instance which returns an open multi-dimensional "meshgrid".
        '''
        pass

    #Building matrices
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diag(self,backend):
        '''
        Extract a diagonal or construct a diagonal array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_diagflat(self,backend):
        '''	
        Create a two-dimensional array with the flattened input as a diagonal.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tri(self,backend):
        '''
        An array with ones at and below the given diagonal and zeros elsewhere.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tril(self,backend):
        '''
        Lower triangle of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_triu(self,backend):
        '''
        Upper triangle of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vander(self,backend):
        '''
        Generate a Vandermonde matrix.
        '''
        pass
    
    #The matrix class
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_bmat(self,backend):
        '''
        Build a matrix object from a string, nested sequence, or array.
        '''
        pass
    
    ######## Array manipulation routines
    # Basic operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_copyto(self,backend):
        '''
        Copies values from one array to another, broadcasting as necessary.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndim(self,backend):
        '''
        Return the number of dimensions of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_shape(self,backend):
        '''
        Return the shape of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_size(self,backend):
        '''
        Return the number of elements along a given axis.
        '''
        pass

    # Changing array shape
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_reshape(self,backend):
        '''
        Gives a new shape to an array without changing its data.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ravel(self,backend):
        '''
        Return a contiguous flattened array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_flat(self,backend):
        '''
        A 1-D iterator over the array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_flatten(self,backend):
        '''
        Return a copy of the array collapsed into one dimension.
        '''
        pass

    # Transpose-like operations
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_moveaxis(self,backend):
        '''
        Move axes of an array to new positions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rollaxis(self,backend):
        '''
        Roll the specified axis backwards, until it lies in a given position.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_swapaxes(self,backend):
        '''
        Interchange two axes of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ndarray_T(self,backend):
        '''
        View of the transposed array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_transpose(self,backend):
        '''
        Returns an array with axes transposed.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_permute_dims(self,backend):
        '''
        Returns an array with axes transposed.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_matrix_transpose(self,backend):
        '''
        Transposes a matrix (or a stack of matrices) x.
        '''
        pass

    # Changing number of dimensions
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_1d(self,backend):
        '''
        Convert inputs to arrays with at least one dimension.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_2d(self,backend):
        '''
        View inputs as arrays with at least two dimensions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_atleast_3d(self,backend):
        '''
        View inputs as arrays with at least three dimensions.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast(self,backend):
        '''
        Produce an object that mimics broadcasting.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast_to(self,backend):
        '''
        Broadcast an array to a new shape.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_broadcast_arrays(self,backend):
        '''
        Broadcast any number of arrays against each other.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_expand_dims(self,backend):
        '''
        Expand the shape of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_squeeze(self,backend):
        '''
        Remove axes of length one from a.
        '''
        pass

    # Changing kind of array
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray(self,backend):
        '''
        Convert the input to an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asanyarray(self,backend):
        '''
        Convert the input to an ndarray, but pass ndarray subclasses through.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asmatrix(self,backend):
        '''
        Interpret the input as a matrix.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asfortranarray(self,backend):
        '''
        Return an array (ndim >= 1) laid out in Fortran order in memory.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_ascontiguousarray(self,backend):
        '''
        Return a contiguous array (ndim >= 1) in memory (C order).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_asarray_chkfinite(self,backend):
        '''
        Convert the input to an array, checking for NaNs or Infs.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_require(self,backend):
        '''
        Return an ndarray of the provided type that satisfies requirements.
        '''
        pass

    # Joining arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_concatenate(self,backend):
        '''
        Join a sequence of arrays along an existing axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_concat(self,backend):
        '''
        Join a sequence of arrays along an existing axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_stack(self,backend):
        '''
        Join a sequence of arrays along a new axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_block(self,backend):
        '''
        Assemble an nd-array from nested lists of blocks.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vstack(self,backend):
        '''
        Stack arrays in sequence vertically (row wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_hstack(self,backend):
        '''
        Stack arrays in sequence horizontally (column wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_dstack(self,backend):
        '''
        Stack arrays in sequence depth wise (along third axis).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_column_stack(self,backend):
        '''
        Stack 1-D arrays as columns into a 2-D array.
        '''
        pass

    # Splitting arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_split(self,backend):
        '''
        Split an array into multiple sub-arrays as views into ary.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_array_split(self,backend):
        '''
        Split an array into multiple sub-arrays.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_dsplit(self,backend):
        '''
        Split array into multiple sub-arrays along the 3rd axis (depth).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_hsplit(self,backend):
        '''
        Split an array into multiple sub-arrays horizontally (column-wise).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_vsplit(self,backend):
        '''
        Split an array into multiple sub-arrays vertically (row-wise).
        '''
        pass

    # Tiling arrays
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_tile(self,backend):
        '''
        Construct an array by repeating A the number of times given by reps.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_repeat(self,backend):
        '''
        Repeat each element of an array after themselves.
        '''
        pass

    # Adding and removing elements
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_delete(self,backend):
        '''
        Return a new array with sub-arrays along an axis deleted.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_insert(self,backend):
        '''
        Insert values along the given axis before the given indices.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_append(self,backend):
        '''
        Append values to the end of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_resize(self,backend):
        '''
        Return a new array with the specified shape.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_trim_zeros(self,backend):
        '''
        Trim the leading and/or trailing zeros from a 1-D array or sequence.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_unique(self,backend):
        '''
        Find the unique elements of an array.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_pad(self,backend):
        '''
        Pad an array.
        '''
        pass

    # Rearranging elements
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flip(self,backend):
        '''
        Reverse the order of elements in an array along the given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_fliplr(self,backend):
        '''
        Reverse the order of elements along axis 1 (left/right).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_flipud(self,backend):
        '''
        Reverse the order of elements along axis 0 (up/down).
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_reshape(self,backend):
        '''
        Gives a new shape to an array without changing its data.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_roll(self,backend):
        '''
        Roll array elements along a given axis.
        '''
        pass

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_rot90(self,backend):
        '''
        Rotate an array by 90 degrees in the plane specified by axes.
        '''
        pass


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", multi_index_data)
    def test_multi_index_matrix(self, backend, data):
        bm.set_backend(backend)
        p = data["p"]
        dim = data["dim"]
        result = data["result"]
        test_result = bm.multi_index_matrix(p, dim)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.multi_index_matrix` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_length(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_length"]
        test_result = bm.edge_length(edge, node)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_length` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_normal(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_normal"]
        test_result = bm.edge_normal(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_normal` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_tangent(self, backend, data):
        bm.set_backend(backend)
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_tangent"]
        test_result = bm.edge_tangent(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                     err_msg=f" `bm.edge_tangent` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_measure(self, backend, data):
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["simple_measure"]
        test_result = bm.simplex_measure(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=15, 
                                     err_msg=f" `bm.simple_measure` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_shape_function(self, backend, data):
        bm.set_backend(backend)
        bcs = bm.from_numpy(data["bcs"])
        p = data["p"]
        result = data["simple_shape_function"]
        test_result = bm.simplex_shape_function(bcs, p)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.simple_shape_function` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", interval_mesh_data)
    def test_interval_grad_lambda(self, backend, data): 
        bm.set_backend(backend)
        line = bm.from_numpy(data["line"])
        node = bm.from_numpy(data["node"])
        result = data["interval_grad_lambda"]
        test_result = bm.interval_grad_lambda(line, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.interval_grad_lambda` function is not equal to real result in backend {backend}")
    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh3d_data)
    def test_triangle_area_3d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_area_3d"]
        test_result = bm.triangle_area_3d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_area_3d` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_triangle_grad_lambda_2d(self, backend, data): 
        bm.set_backend(backend)
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_grad_lambda_2d"]
        test_result = bm.triangle_grad_lambda_2d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_grad_lambda_2d` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
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
    pytest.main(['test_backends.py', '-q'])
