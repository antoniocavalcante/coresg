import numpy as np
cimport numpy as np

cimport cython

include '../../parameters.pxi'

cdef class FairSplitTree:

    cdef FairSplitTreeNode root

    cdef void construct(
        self, 
        np.ndarray[DTYPE_t, ndim=2] data, 
        np.ndarray[DTYPE_t, ndim=1] core_distances)


cdef class FairSplitTreeNode:

    cdef np.ndarray points
    cdef FairSplitTreeNode l, r
    cdef np.float64_t diameter
    cdef np.ndarray center
    cdef bint leaf

cdef DTYPE_t euclidean(np.ndarray[np.float64_t, ndim=1] v1, np.ndarray[np.float64_t, ndim=1] v2)