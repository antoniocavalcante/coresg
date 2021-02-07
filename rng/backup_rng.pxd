import numpy as np
cimport numpy as np

cimport cython

from rng.fair_split_tree cimport FairSplitTree, FairSplitTreeNode, separated

include '../parameters.pxi'

cdef class RelativeNeighborhoodGraph:

    cdef bint quick, naive
    cdef ITYPE_t min_points, n
    cdef list u, v, w

    cpdef graph(self)

    cdef void wspd(
        self, 
        FairSplitTree fst,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)
        
    cdef void find_pairs(
        self, 
        FairSplitTreeNode node_a,
        FairSplitTreeNode node_b,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)

    cdef void sbcn(self, 
        np.ndarray[ITYPE_t, ndim=1] red, 
        np.ndarray[ITYPE_t, ndim=1] blue,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)

    cdef void add_edge(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight, 
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)    

    cdef bint relative_neighbors(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)

    cdef bint _relative_neighbors_quick(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn)

    cdef bint _relative_neighbors_naive(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances)

    cdef DTYPE_t _mutual_reachability_distance(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b,
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=1] core_distances)


cdef DTYPE_t euclidean_local(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2)