"""
METIS for Python
================

Wrapper for the METIS library for partitioning graphs (and other stuff).

This library is unrelated to PyMetis, except that they wrap the same library.
PyMetis is a Boost Python extension, while this library is pure python and will
run under PyPy and interpreters with similarly compatible ctypes libraries.

NetworkX_ is recommended for representing graphs for use with this wrapper,
but it isn't required. Simple adjacency lists are supported as well.

.. _NetworkX: http://networkx.lanl.gov/

The function of primary interest in this module is :func:`part_graph`.

Other objects in the module may be of interest to those looking to
mangle their graph datastructures into the required format. Examples
of this include the :func:`networkx_to_metis` and :func:`adjlist_to_metis` functions.
These are automatically called by :func:`part_graph`, so there is
little need to call them manually.

See the BitBucket repository_  for updates and issue reporting.

.. _repository: http://bitbucket.org/kw/metis-python

Installation
============

It's on PyPI, so installation should be as easy as::

    pip install metis
          -or-
    easy_install metis

METIS itself is not included with this wrapper. Get it here_.

.. _here: http://glaros.dtc.umn.edu/gkhome/views/metis

Note that the shared library is needed, and isn't enabled by default
by the configuration process. Turn it on by issuing::

    make config shared=1

Your operating system's package manager might know about METIS,
but this wrapper was designed for use with METIS 5. Packages with
METIS 4 will not work.

This wrapper uses a few environment variables:

.. envvar:: METIS_DLL

    This wrapper uses Python's ctypes module to interface with the METIS
    shared library. If it is unable to automatically locate the library, you
    may specify the full path to the library file in this environment variable.

.. envvar:: METIS_IDXTYPEWIDTH
.. envvar:: METIS_REALTYPEWIDTH

    The sizes of the :c:type:`idx_t` and :c:type:`real_t` types are not
    easily determinable at runtime, so they can be provided with these
    environment variables. The default value for each of these (at both compile
    time and in this library) is 32, but they may be set to 64 if desired. If
    the values do not match what was used to compile the library, Bad Things(TM)
    will occur.

Example
=======

    >>> import networkx as nx
    >>> import metis
    >>> G = metis.example_networkx()
    >>> (edgecuts, parts) = metis.part_graph(G, 3)
    >>> colors = ['red','blue','green']
    >>> for i, p in enumerate(parts):
    ...     G.node[i]['color'] = colors[p]
    ...
    >>> nx.write_dot(G, 'example.dot') # Requires pydot or pygraphviz

.. graphviz:: example.dot

"""
# Copyright (c) 2012 Ken Watford
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# tl;dr - MIT license.

__version__ = '0.2a1'

import ctypes
from ctypes import POINTER as P, byref
import os, sys, operator as op
from warnings import warn
from collections import namedtuple
import numpy as np
try:
    import networkx
except ImportError:
    networkx = None

__all__ = ['part_graph', 'networkx_to_metis', 'adjlist_to_metis']

# Sadly, METIS does not currently include any API call to determine
# the correct datatypes. So we either have to guess, let the user tell
# us, try to infer it by checking API behavior on test inputs, or
# look for the header and parse out the preprocessor macros.
# Since we're in a bit of a hurry, for now we'll use the defaults
# and let the user specify if this is wrong with env vars.
IDXTYPEWIDTH  = os.getenv('METIS_IDXTYPEWIDTH', '32')
REALTYPEWIDTH = os.getenv('METIS_REALTYPEWIDTH', '32')

if IDXTYPEWIDTH == '32':
    idx_t = ctypes.c_int32
elif IDXTYPEWIDTH == '64':
    idx_t = ctypes.c_int64
else:
    raise EnvironmentError('Env var METIS_IDXTYPEWIDTH must be "32" or "64"')

if REALTYPEWIDTH == '32':
    real_t = ctypes.c_float
elif REALTYPEWIDTH == '64':
    real_t = ctypes.c_double
else:
    raise EnvironmentError('Env var METIS_REALTYPEWIDTH must be "32" or "64"')


METIS_NOPTIONS = 40

# The _enum and _bitfield base classes come from my PyCL project
# They make enum constants a little more friendly.
class _enum(ctypes.c_int32):
    # Base class for various enums
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value
    def __ne__(self, other):
        return not(self == other)
    def __hash__(self):
        return self.value.__hash__()
    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        else:
            return "UNKNOWN(0x%x)" % self.value
    def shortname(self):
        return self._by_value_short.get(self, 'unknown')
    @classmethod
    def fromname(cls, name):
        if name in cls._by_name:
            return cls._by_name[name]
        elif name in cls._by_name_short:
            return cls._by_name_short[name]
        else:
            raise KeyError('Unknown name: %s' % name)
    @classmethod
    def toname(cls, val):
        if val in self._by_value_short:
            return cls._by_value_short[value]
        elif val in cls._by_value:
            return cls._by_value[value]
        else:
            raise KeyError

class _bitfield(ctypes.c_int32):
    # Base class for bitfield values
    # Bitwise operations for combining flags are supported.
    def __or__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value | other.value)
    def __and__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value & other.value)
    def __xor__(self, other):
        assert isinstance(other, (int, self.__class__))
        return self.__class__(self.value ^ other.value)
    def __not__(self):
        return self.__class__(~self.value)
    def __contains__(self, other):
        assert isinstance(other, self.__class__)
        return (self.value & other.value) == other.value
    def __hash__(self):
        return self.value.__hash__()
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        for val in by_value:
            if val in self:
                names.append(by_value[val])
        if names:
            return " | ".join(names)
        else:
            return "UNKNOWN(0x%x)" % self.value

class rstatus_et(_enum):
    METIS_OK              =  1    #/*!< Returned normally */
    METIS_ERROR_INPUT     = -2   #/*!< Returned due to erroneous inputs and/or options */
    METIS_ERROR_MEMORY    = -3   #/*!< Returned due to insufficient memory */
    METIS_ERROR           = -4   #/*!< Some other errors */

class moptype_et(_enum):
    METIS_OP_PMETIS = 0
    METIS_OP_KMETIS = 1
    METIS_OP_OMETIS = 2

class moptions_et(_enum):
    METIS_OPTION_PTYPE     =  0
    METIS_OPTION_OBJTYPE   =  1
    METIS_OPTION_CTYPE     =  2
    METIS_OPTION_IPTYPE    =  3
    METIS_OPTION_RTYPE     =  4
    METIS_OPTION_DBGLVL    =  5
    METIS_OPTION_NITER     =  6
    METIS_OPTION_NCUTS     =  7
    METIS_OPTION_SEED      =  8
    METIS_OPTION_NO2HOP    =  9
    METIS_OPTION_MINCONN   = 10
    METIS_OPTION_CONTIG    = 11
    METIS_OPTION_COMPRESS  = 12
    METIS_OPTION_CCORDER   = 13
    METIS_OPTION_PFACTOR   = 14
    METIS_OPTION_NSEPS     = 15
    METIS_OPTION_UFACTOR   = 16
    METIS_OPTION_NUMBERING = 17
    #/* Used for command-line parameter purposes */
    METIS_OPTION_HELP      = 17
    METIS_OPTION_TPWGTS    = 18
    METIS_OPTION_NCOMMON   = 19
    METIS_OPTION_NOOUTPUT  = 20
    METIS_OPTION_BALANCE   = 21
    METIS_OPTION_GTYPE     = 22
    METIS_OPTION_UBVEC     = 23

class mptype_et(_enum):
    METIS_PTYPE_DEFAULT = -1
    METIS_PTYPE_RB   = 0
    METIS_PTYPE_KWAY = 1

class mgtype_et(_enum):
    METIS_GTYPE_DEFAULT = -1
    METIS_GTYPE_DUAL  = 0
    METIS_GTYPE_NODAL = 1

class mctype_et(_enum):
    METIS_CTYPE_DEFAULT = -1
    METIS_CTYPE_RM   = 0
    METIS_CTYPE_SHEM = 1

class miptype_et(_enum):
    METIS_IPTYPE_DEFAULT = -1
    METIS_IPTYPE_GROW    = 0
    METIS_IPTYPE_RANDOM  = 1
    METIS_IPTYPE_EDGE    = 2
    METIS_IPTYPE_NODE    = 3
    METIS_IPTYPE_METISRB = 4

class mrtype_et(_enum):
    METIS_RTYPE_DEFAULT   = -1
    METIS_RTYPE_FM        = 0
    METIS_RTYPE_GREEDY    = 1
    METIS_RTYPE_SEP2SIDED = 2
    METIS_RTYPE_SEP1SIDED = 3

class mdbglvl_et(_bitfield):
    METIS_DBG_DEFAULT    = -1
    METIS_DBG_INFO       = 1       #/*!< Shows various diagnostic messages */
    METIS_DBG_TIME       = 2       #/*!< Perform timing analysis */
    METIS_DBG_COARSEN    = 4       #/*!< Show the coarsening progress */
    METIS_DBG_REFINE     = 8       #/*!< Show the refinement progress */
    METIS_DBG_IPART      = 16      #/*!< Show info on initial partitioning */
    METIS_DBG_MOVEINFO   = 32      #/*!< Show info on vertex moves during refinement */
    METIS_DBG_SEPINFO    = 64      #/*!< Show info on vertex moves during sep refinement */
    METIS_DBG_CONNINFO   = 128     #/*!< Show info on minimization of subdomain connectivity */
    METIS_DBG_CONTIGINFO = 256     #/*!< Show info on elimination of connected components */
    METIS_DBG_MEMORY     = 2048    #/*!< Show info related to wspace allocation */
    METIS_DBG_ALL        = sum(2**i for i in list(range(9))+[11])

class mobjtype_et(_enum):
    METIS_OBJTYPE_DEFAULT = -1
    METIS_OBJTYPE_CUT  = 0
    METIS_OBJTYPE_VOL  = 1
    METIS_OBJTYPE_NODE = 2

# For enums and bitfields, do magic. Each type gets a registry of the
# names and values of their defined elements, to support pretty printing.
# Further, each of the class variables (which are defined using ints) is
# upgraded to be a member of the class in question.
# Additionally, each of the constants is copied into the module scope.
for cls in (_enum.__subclasses__() + _bitfield.__subclasses__()):
    if cls.__name__ not in globals() or cls.__name__.startswith('_'):
        # Don't apply this to types that ctypes makes automatically,
        # like the _be classes. Doing so will overwrite the declared
        # constants at global scope, which is really weird.
        continue
    cls._by_name = dict()
    cls._by_value = dict()
    cls._by_name_short = dict()
    cls._by_value_short = dict()
    if not cls.__doc__:
        cls.__doc__ = ""
    for name, value in cls.__dict__.items():
        if isinstance(value, int):
            obj = cls(value)
            setattr(cls, name, obj)
            cls._by_name[name] = obj
            cls._by_value[obj] = name
            shortname = name.split('_')[-1].lower()
            cls._by_name_short[shortname] = obj
            cls._by_value_short[obj] = shortname
            globals()[name] = obj
            cls.__doc__ += """
            .. attribute:: %s
            """ % name
# cleanup
del cls; del name; del value; del obj


# Convert values taken from option array into appropriate enum
_opt_types = {
    METIS_OPTION_PTYPE   : mptype_et,
    METIS_OPTION_OBJTYPE : mobjtype_et,
    METIS_OPTION_CTYPE   : mctype_et,
    METIS_OPTION_GTYPE   : mgtype_et,
    METIS_OPTION_IPTYPE  : miptype_et,
    METIS_OPTION_RTYPE   : mrtype_et,
    METIS_OPTION_DBGLVL  : mdbglvl_et,
    }

class METIS_Options(object):
    """ Represents the 'options' array used to represent all
    nearly all options that can be given to METIS functions.
    Will be used when extra keyword arguments are are used in wrappers.

    Note that I spent way too much time on this.

    """
    def __init__(self, options=None, **opts):
        self.array = (idx_t*METIS_NOPTIONS)()
        _METIS_SetDefaultOptions(self.array)
        if options:
            for opt, val in options.keys():
                self[opt] = val
        for opt, val in opts.items():
            self[opt] = val

    def keys(self):
        return moptions_et._by_name_short.keys()

    def __getitem__(self, opt):
        if isinstance(opt, str):
            opt = moptions_et.fromname(opt)
        val = self.array[opt.value]
        if opt in _opt_types:
            val = _opt_types[opt](val)
            if isinstance(val, _enum):
                val = val.shortname()
        return val

    def __setitem__(self, opt, val):
        if isinstance(opt, str):
            opt = moptions_et.fromname(opt)
        if isinstance(val, str) and opt in _opt_types:
            val = _opt_types[opt].fromname(val)
        try:
            self.array[opt.value] = val
        except TypeError:
            raise TypeError("Bad type for option %s: %s" %
                (opt, val.__class__.__name__))

    def __repr__(self):
        """ Only show non-default options """
        nondefaults = []
        for opt in self.keys():
            realind = moptions_et.fromname(opt).value
            if self.array[realind] != -1:
                val = self[opt]
                nondefaults.append('%s=%r' % (opt, val))
        return 'METIS_Options(' + ', '.join(nondefaults) + ')'



# Attempt to locate and load the appropriate shared library
_dll_filename = os.getenv('METIS_DLL')
if not _dll_filename:
    try:
        from ctypes.util import find_library as _find_library
        _dll_filename = _find_library('metis')
    except ImportError:
        pass
if _dll_filename == 'SKIP':
    warn('$METIS_DLL=SKIP, skipping DLL load. Nothing will work. '
         'This is normal during install.', UserWarning, 2)
    _dll = None
elif _dll_filename:
    try:
        _dll = ctypes.cdll.LoadLibrary(_dll_filename)
    except:
        raise RuntimeError('Could not load METIS dll: %s' % _dll_filename)
else:
    if os.environ.get('READTHEDOCS', None) == 'True':
        # Don't care if we can load the DLL on RTD.
        _dll = None
    else:
        raise RuntimeError('Could not locate METIS dll. Please set the METIS_DLL environment variable to its full path.')

# Wrapping conveniences

def _wrapdll(*argtypes, **kw):
    """
    Decorator used to simplify wrapping METIS functions a bit.

    The positional arguments represent the ctypes argument types the
    C-level function expects, and will be used to do argument type checking.

    If a `res` keyword argument is given, it represents the C-level
    function's expected return type. The default is `rstatus_et`

    If an `err` keyword argument is given, it represents an error checker
    that should be run after low-level calls. The `_result_errcheck` and
    `_lastarg_errcheck` functions should be sufficient for most OpenCL
    functions. `_result_errcheck` is the default value.

    The decorated function should have the same name as the underlying
    METIS function, since the function name is used to do the lookup. The
    C-level function pointer will be stored in the decorated function's
    `call` attribute, and should be used by the decorated function to
    perform the actual call(s). The wrapped function is otherwise untouched.

    """
    def dowrap(f):
        if f.__name__.startswith('_'):
            name = f.__name__[1:]
        else:
            name = f.__name__
        if _dll:
            wrapped_func = getattr(_dll, name)
            wrapped_func.argtypes = argtypes
            res = kw.pop('res', rstatus_et)
            wrapped_func.restype = res
            err = kw.pop('err', _result_errcheck)
            wrapped_func.errcheck = err
            f.call = wrapped_func
        else:
            def nodll(*args, **kw):
                raise NotImplemented("No METIS DLL")
            f.call = nodll
        return f
    return dowrap

# Translate METIS status messages into Python exceptions
class METIS_Error(Exception): pass
class METIS_MemoryError(METIS_Error, MemoryError): pass
class METIS_InputError(METIS_Error, ValueError): pass
class METIS_OtherError(METIS_Error): pass

def _result_errcheck(result, func, args):
    """
    For use in the errcheck attribute of a ctypes function wrapper.

    Most METIS functions return rstatus_et. This checks it for
    an error code and raises an appropriate exception if it finds one.

    This is the default error checker when using _wrapdll
    """
    if result != METIS_OK:
        if result == METIS_ERROR_INPUT: raise METIS_InputError
        if result == METIS_ERROR_MEMORY: raise METIS_MemoryError
        if result == METIS_ERROR: raise METIS_OtherError
        raise RuntimeError("Error raising error: Bad error.") # lolwut
    return result

# Graph helpers

METIS_Graph = namedtuple('METIS_Graph',
    'nvtxs ncon xadj adjncy vwgt vsize adjwgt')

def networkx_to_metis(G):
    """
    Convert NetworkX graph into something METIS can consume
    The graph may specify weights and sizes using the following
    graph attributes:

    * ``edge_weight_attr``
    * ``node_weight_attr`` (multiple names allowed)
    * ``node_size_attr``

    For example::

        >>> G.edge[0][1]['weight'] = 3
        >>> G.node[0]['quality'] = 5
        >>> G.node[0]['specialness'] = 8
        >>> G.graph['edge_weight_attr'] = 'weight'
        >>> G.graph['node_weight_attr'] = ['quality', 'specialness']

    If node_weight_attr is a list instead of a string, then multiple
    node weight labels can be provided.

    All weights must be integer values. If an attr label is specified but
    a node/edge is missing that attribute, it defaults to 1.

    If a graph attribute is not provided, no defaut is used. That is, if
    ``edge_weight_attr`` is not set, then ``'weight'`` is not used as the
    default, and the graph will appear unweighted to METIS.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    nvtxs = idx_t(n)

    H = networkx.convert_node_labels_to_integers(G)
    xadj = (idx_t*(n+1))()
    adjncy = (idx_t*(2*m))()

    # Check graph attributes for weight/size labels
    edgew = G.graph.get('edge_weight_attr', None)
    nodew = G.graph.get('node_weight_attr', [])
    nodesz = G.graph.get('node_size_attr', None)

    if edgew:
        adjwgt = (idx_t*(2*m))()
    else:
        adjwgt = None

    if nodew:
        if isinstance(nodew, str):
            nodew = [nodew]
        nc = len(nodew)
        ncon = idx_t(nc)
        vwgt = (idx_t*(n*len(nodew)))()
    else:
        ncon = idx_t(1)
        vwgt = None

    if nodesz:
        vsize = (idx_t*n)()
    else:
        vsize = None

    # Fill in each array
    xadj[0] = e = 0
    for i in H.node:
        for c,w in enumerate(nodew):
            try:
                vwgt[i*nc+c] = H.node[i].get(w, 1)
            except TypeError:
                raise TypeError("Node weights must be integers" )

        if nodesz:
            try:
                vsize[i] = H.node[i].get(nodesz, 1)
            except TypeError:
                raise TypeError("Node sizes must be integers")

        for j, attr in H.edge[i].items():
            adjncy[e] = j
            if edgew:
                try:
                    adjwgt[e] = attr.get(edgew, 1)
                except TypeError:
                    raise TypeError("Edge weights must be integers")
            e += 1
        xadj[i+1] = e

    return METIS_Graph(nvtxs, ncon, xadj, adjncy, vwgt, vsize, adjwgt)

def adjlist_to_metis(adjlist, nodew=None, nodesz=None):
    """
    Rudimentary adjacency list converter.
    Primarily of use if you don't have or don't want to use NetworkX.

    :param adjlist: A list of tuples. Each list element represents a node or vertex
      in the graph. Each item in the tuples represents an edge. These items may be
      single integers representing neighbor index, or they may be an (index, weight)
      tuple if you want weighted edges. Default weight is 1 for missing weights.

      The graph must be undirected, and each edge must be represented twice (once for
      each node). The weights should be identical, if provided.
    :param nodew: is a list of node weights, and must be the same size as `adjlist` if given.
      If desired, the elements of `nodew` may be tuples of the same size (>= 1) to provided
      multiple weights for each node.
    :param nodesz: is a list of node sizes. These are relevant when doing volume-based
      partitioning.

    Note that all weights and sizes must be non-negative integers.
    """
    n = len(adjlist)
    m2 = sum(map(len, adjlist))

    xadj = (idx_t*(n+1))()
    adjncy = (idx_t*m2)()
    adjwgt = (idx_t*m2)()
    seen_adjwgt = False # Don't use adjwgt unless we've seen any

    ncon = idx_t(1)
    if nodew:
        if isinstance(nodew[0], int):
            vwgt = (idx_t*n)(*nodew)
        else: # Assume a list of them
            nw = len(nodew[0])
            ncon = idx_t(nw)
            vwgt = (idx_t*(nw*n))(*reduce(op.add,nodew))
    else:
        vwgt = None

    if nodesz:
        vsize = (idx_t*n)(*nodesz)
    else:
        vsize = None

    xadj[0] = e = 0
    for i, adj in enumerate(adjlist):
        for j in adj:
            try:
                adjncy[e], adjwgt[e] = j
                seen_adjwgt = True
            except TypeError:
                adjncy[e], adjwgt[e] = j, 1
            e += 1
        xadj[i+1] = e

    if not seen_adjwgt:
        adjwgt = None

    return METIS_Graph(idx_t(n), ncon, xadj, adjncy, vwgt, vsize, adjwgt)


def array_to_metis(adj, adjLocation, nodew=None, nodesz=None):
    """
    Rudimentary adjacency list converter.
    Primarily of use if you don't have or don't want to use NetworkX.

    :param mesh: 
    :param nodew: 
    :param nodesz:

    Note that all weights and sizes must be non-negative integers.
    """
    n = len(adjLocation) - 1
    m2 = adjLocation[-1] 

    if adj.dtype == 'int32':
        if idx_t == ctypes.POINTER(ctypes.c_int64):
            adj = adj.astype(np.int64)
            adjLocation = adjLocation.astype(np.int64)

            adjncy = adj.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            xadj = adjLocation.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)) 
        else:
            adjncy = adj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            xadj = adjLocation.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) 
    elif adj.dtype == 'int64':
        if idx_t == ctypes.POINTER(ctypes.c_int32):
            raise TypeError
        adjncy = adj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        xadj = adjLocation.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) 
    else:
        raise TypeError


    adjwgt = None 
    ncon = idx_t(1)
    if nodew:
        if isinstance(nodew[0], int):
            vwgt = (idx_t*n)(*nodew)
        else: # Assume a list of them
            nw = len(nodew[0])
            ncon = idx_t(nw)
            vwgt = (idx_t*(nw*n))(*reduce(op.add,nodew))
    else:
        vwgt = None

    if nodesz:
        vsize = (idx_t*n)(*nodesz)
    else:
        vsize = None

    return METIS_Graph(idx_t(n), ncon, xadj, adjncy, vwgt, vsize, adjwgt)


### Wrapped METIS functions ###

@_wrapdll(P(idx_t))
def _METIS_SetDefaultOptions(optarray):
    _METIS_SetDefaultOptions.call(optarray)

@_wrapdll(P(idx_t), P(idx_t), P(idx_t), P(idx_t),
    P(idx_t), P(idx_t), P(idx_t), P(idx_t), P(real_t),
    P(real_t), P(idx_t), P(idx_t), P(idx_t))
def _METIS_PartGraphKway(nvtxs, ncon, xadj, adjncy, vwgt, vsize,
                adjwgt, nparts, tpwgts, ubvec, options, objval, part):
    """
    Called by `part_graph`
    """
    return _METIS_PartGraphKway.call(nvtxs, ncon, xadj, adjncy, vwgt, vsize,
                adjwgt, nparts, tpwgts, ubvec, options, objval, part)

@_wrapdll(P(idx_t), P(idx_t), P(idx_t), P(idx_t),
    P(idx_t), P(idx_t), P(idx_t), P(idx_t), P(real_t),
    P(real_t), P(idx_t), P(idx_t), P(idx_t))
def _METIS_PartGraphRecursive(nvtxs, ncon, xadj, adjncy, vwgt, vsize,
                adjwgt, nparts, tpwgts, ubvec, options, objval, part):
    """
    Called by `part_graph`
    """
    return _METIS_PartGraphRecursive.call(nvtxs, ncon, xadj, adjncy, vwgt, vsize,
                adjwgt, nparts, tpwgts, ubvec, options, objval, part)

### End METIS wrappers. ###

def part_mesh(mesh, entity='cell', nparts=2, 
        tpwgts=None, ubvec=None, recursive=False, **opts):
    """ Perform graph partitioning using k-way or recursive methods

    Return ....

    :param mesh: a mesh 
    
    """
    if entity == 'cell':
        adj, adjLocation = mesh.ds.cell_to_cell(return_array=True)
    elif entity == 'node':
        adj, adjLocation = mesh.ds.node_to_node(return_array=True)

    graph = array_to_metis(adj, adjLocation)

    options = METIS_Options(**opts)
    print(options)
    print(options.keys());
    if tpwgts and not isinstance(tpwgts, ctypes.Array):
        if isinstance(tpwgts[0], (tuple, list)):
            tpwgts = reduce(op.add, tpwgts)
        tpwgts = (real_t*len(tpwgts))(*tpwgts)
    if ubvec and not isinstance(ubvec, ctypes.Array):
        ubvec = (real_t*len(ubvect))(*ubvec)

    if tpwgts: assert len(tpwgts) == nparts * graph.ncon
    if ubvec: assert len(ubvec) == graph.ncon

    nparts_var = idx_t(nparts)

    objval = idx_t()
    partition = (idx_t*graph.nvtxs.value)()

    args = (byref(graph.nvtxs), byref(graph.ncon), graph.xadj,
            graph.adjncy, graph.vwgt, graph.vsize, graph.adjwgt,
            byref(nparts_var), tpwgts, ubvec, options.array,
            byref(objval), partition)
    if recursive:
        _METIS_PartGraphRecursive(*args)
    else:
        _METIS_PartGraphKway(*args)

    return objval.value, np.ctypeslib.as_array(partition)

def part_graph(graph, nparts=2,
    tpwgts=None, ubvec=None, recursive=False, **opts):
    """
    Perform graph partitioning using k-way or recursive methods.

    Returns a 2-tuple `(objval, parts)`, where `parts` is a list of
    partition indices corresponding and `objval` is the value of
    the objective function that was minimized (either the edge cuts
    or the total volume).

    :param graph: may be a NetworkX graph, an adjacency list, or a :class:`METIS_Graph`
      named tuple. To use the named tuple approach, you'll need to
      read the METIS manual for the meanings of the fields.

      See :func:`networkx_to_metis` for help and details on how the
      graph is converted and how node/edge weights and sizes can
      be specified.

      See :func:`adjlist_to_metis` for information on the use of adjacency lists.
      The extra ``nodew`` and ``nodesz`` keyword arguments of that function may be given
      directly to this function and will be forwarded to the converter.
      Alternatively, a dictionary can be provided as ``graph`` and its items
      will be passed as keyword arguments.
    :param nparts: The target number of partitions. You might get fewer.
    :param tpwgts: Target partition weights. For each partition, there should
      be one (float) weight for each node constraint. That is, if `nparts` is 3 and
      each node of the graph has two weights, then tpwgts might look like this::

        [(0.5, 0.1), (0.25, 0.8), (0.25, 0.1)]

      This list may be provided flattened. The internal tuples are for convenience.
      The partition weights for each constraint must sum to 1.
    :param ubvec: The load imalance tolerance for each node constraint. Should be
      a list of floating point values each greater than 1.

    :param recursive: Determines whether the partitioning should be done by
      direct k-way cuts or by a series of recursive cuts. These correspond to
      :c:func:`METIS_PartGraphKway` and :c:func:`METIS_PartGraphRecursive` in
      METIS's C API.

    Any additional METIS options may be specified as keyword parameters.

    For k-way clustering, the appropriate options are::

        objtype   = 'cut' or 'vol'
        ctype     = 'rm' or 'shem'
        iptype    = 'grow', 'random', 'edge', 'node'
        rtype     = 'fm', 'greedy', 'sep2sided', 'sep1sided'
        ncuts     = integer, number of cut attempts (default = 1)
        niter     = integer, number of iterations (default = 10)
        ufactor   = integer, maximum load imbalance of (1+x)/1000
        minconn   = bool, minimize degree of subdomain graph
        contig    = bool, force contiguous partitions
        seed      = integer, RNG seed
        numbering = 0 (C-style) or 1 (Fortran-style) indices
        dbglvl    = Debug flag bitfield

    For recursive clustering, the appropraite options are::

        ctype     = 'rm' or 'shem'
        iptype    = 'grow', 'random', 'edge', 'node'
        rtype     = 'fm', 'greedy', 'sep2sided', 'sep1sided'
        ncuts     = integer, number of cut attempts (default = 1)
        niter     = integer, number of iterations (default = 10)
        ufactor   = integer, maximum load imbalance of (1+x)/1000
        seed      = integer, RNG seed
        numbering = 0 (C-style) or 1 (Fortran-style) indices
        dbglvl    = Debug flag bitfield

    See the METIS manual for specific meaning of each option.
    """

    if networkx and isinstance(graph, networkx.Graph):
        graph = networkx_to_metis(graph)
    elif isinstance(graph, list):
        nodesz = opts.pop('nodesz', None)
        nodew  = opts.pop('nodew', None)
        graph = adjlist_to_metis(graph, nodew, nodesz)
    elif isinstance(graph, dict):
        # Check if this has METIS_Graph fields or an adjlist
        if 'nvtxs' in graph:
            graph = METIS_Graph(**graph)
        elif 'adjlist' in graph:
            graph = adjlist_to_metis(**graph)

    options = METIS_Options(**opts)
    if tpwgts and not isinstance(tpwgts, ctypes.Array):
        if isinstance(tpwgts[0], (tuple, list)):
            tpwgts = reduce(op.add, tpwgts)
        tpwgts = (real_t*len(tpwgts))(*tpwgts)
    if ubvec and not isinstance(ubvec, ctypes.Array):
        ubvec = (real_t*len(ubvect))(*ubvec)

    if tpwgts: assert len(tpwgts) == nparts * graph.ncon
    if ubvec: assert len(ubvec) == graph.ncon

    nparts_var = idx_t(nparts)

    objval = idx_t()
    partition = (idx_t*graph.nvtxs.value)()

    args = (byref(graph.nvtxs), byref(graph.ncon), graph.xadj,
            graph.adjncy, graph.vwgt, graph.vsize, graph.adjwgt,
            byref(nparts_var), tpwgts, ubvec, options.array,
            byref(objval), partition)
    if recursive:
        _METIS_PartGraphRecursive(*args)
    else:
        _METIS_PartGraphKway(*args)

    return objval.value, list(partition)

def example_adjlist():
    return [[1, 2, 3, 4], [0], [0], [0], [0, 5], [4, 6], [13, 5, 7],
            [8, 6], [9, 10, 11, 12, 7], [8], [8], [8], [8], [14, 6], [13, 15],
            [16, 17, 18, 14], [15], [15], [15]]

def example_networkx():
    G = networkx.Graph()
    G.add_star([0,1,2,3,4])
    G.add_path([4,5,6,7,8])
    G.add_star([8,9,10,11,12])
    G.add_path([6,13,14,15])
    G.add_star([15,16,17,18])
    return G

def test():
    adjlist = example_adjlist()

    print("Testing k-way cut")
    cuts, parts = part_graph(adjlist, 3, recursive=False, dbglvl=METIS_DBG_ALL)
    assert cuts == 2
    assert set(parts) == set([0,1,2])

    print("Testing recursive cut")
    cuts, parts = part_graph(adjlist, 3, recursive=True, dbglvl=METIS_DBG_ALL)
    assert cuts == 2
    assert set(parts) == set([0,1,2])

    if networkx:
        print("Testing with NetworkX")
        G = example_networkx()
        cuts, parts = part_graph(G, 3)
        assert cuts == 2
        assert set(parts) == set([0,1,2])

    print("METIS appears to be working.")

if __name__ == '__main__':
    test()
