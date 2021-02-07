import numpy as np
cimport numpy as np

cdef class FairSplitTree:
    cdef FairSplitTreeNode root

    cdef construct(
        self, 
        np.ndarray[np.float64_t, ndim=2] data, 
        np.ndarray[np.float64_t, ndim=1] core_distances)

cpdef bint separated(FairSplitTreeNode node_a, FairSplitTreeNode node_b)

cdef np.float64_t node_distances(FairSplitTreeNode node_a, FairSplitTreeNode node_b)

cdef np.float64_t euclidean(np.ndarray[np.float64_t, ndim=1] v1, np.ndarray[np.float64_t, ndim=1] v2)

cdef class FairSplitTreeNode:
    cdef public np.ndarray points
    cdef public FairSplitTreeNode l, r
    cdef public np.float64_t diameter
    cdef public np.ndarray center
    cdef public bint leaf