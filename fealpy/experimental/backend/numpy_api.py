import numpy as np


constants = [
        "e", 
        "euler_gamma",
        "inf",
        "nan", 
        "newaxis",
        "pi",
        ]

array_creation = [
        # from shape or value
        "empty",
        "empty_like", 
        "eye",
        "identity",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
        # from existing data
        "array",
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asmatrix",
        "astype",
        "copy",
        "frombuffer",
        "from_dlpack",
        "from_file",
        "from_function",
        "fromiter",
        "fromstring",
        "loadtxt",
        # numerical ranges
        "arange",
        "linspace",
        "logspace",
        "geomspace",
        "meshgrid",
        ]

array_manipulation = [
        # basic operations
        "copyto",
        "ndim",
        "shape",
        "size",
        # changing array shape
        "reshape",
        "ravel",
        # transpose-like operations
        "moveaxis", 
        "rollaxis",
        "swapaxes",
        "transpose",
        "permute_dims",
        # change number of dimensions
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "broadcast",
        "broadcast_to",
        "broadcast_arrays",
        "expand_dims",
        "squeeze",
        # joining arrays
        "concatenate",
        "concat",
        "stack",
        "block",
        "vstack",
        "hstack",
        "dstack",
        "column_stack"
        # splitting arrays
        "split",
        "array_split",
        "dsplit",
        "hsplit",
        "vsplit",
        # tiling arrays
        "tile",
        "repeat",
        # adding and removing elements
        "delete",
        "insert",
        "append",
        "resize",
        "trim_zeros",
        "unique",
        "pad",
        # rearranging elements
        "flip", 
        "fliplr",
        "roll",
        ]

bitwise_operations = [
        # elementwise bit operations
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "invert",
        "bitwise_invert",
        "left_shift",
        "bitwise_left_shit",
        "right_shift",
        "bitwise_right_shift",
        # bit packing
        "packbits",
        "unpackbits",
        "binary_repr"
        ]

data_type_routines = [
        "can_cast",
        "promote_types",
        "min_scalar_type",
        "result_type",
        "common_type",
        # creating data types
        "dtype",
        # data type information
        "finfo",
        "iinfo",
        # data type testing
        "isdtype",
        "issubdtype",
        # data type testing
        "typename",
        "mintypecode"
        ]
