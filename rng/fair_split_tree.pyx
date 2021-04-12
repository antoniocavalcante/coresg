import math

import numpy as np
cimport numpy as np

cimport cython

from scipy.spatial import distance

from libc.math cimport sqrt

include '../parameters.pxi'

cdef class FairSplitTree:

    cdef public FairSplitTreeNode root

    def __init__(
        self, 
        np.ndarray[DTYPE_t, ndim=2] data, 
        np.ndarray[DTYPE_t, ndim=1] core_distances):
 
        self.root = FairSplitTreeNode(np.arange(len(data), dtype=ITYPE))

        self.construct(data, core_distances)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef construct(
        self, 
        np.ndarray[DTYPE_t, ndim=2] data, 
        np.ndarray[DTYPE_t, ndim=1] core_distances):

        cdef FairSplitTreeNode node
        
        cdef ITYPE_t point, split_dim, i, idx_left, idx_right, n, current_idx
        cdef np.ndarray left, right
        cdef DTYPE_t split_val
        cdef np.ndarray[DTYPE_t, ndim=1] maxdim, mindim

        # starts with the root of the tree
        cdef list stack = [self.root]

        current_idx = 1

        left  = np.zeros(len(data), dtype=ITYPE)
        right = np.zeros(len(data), dtype=ITYPE)

        while stack:
            
            node = stack.pop()

            n = len(node.points)

            if n > 1:
                # finds the maximum and minimum values for each dimension
                maxdim = np.amax(data[node.points], axis=0)
                mindim = np.amin(data[node.points], axis=0)

                # updates the diameter of the node (euclidean distance)
                node.diameter = max(
                    euclidean(maxdim, mindim),
                    np.max(core_distances[node.points]))

                # updates the geometric center of this node
                node.center = (maxdim + mindim)/2

                split_dim = np.argmax(maxdim - mindim)
                split_val = (mindim[split_dim] + maxdim[split_dim]) / 2
                
                idx_left  = 0
                idx_right = 0

                # left  = np.zeros(n, dtype=ITYPE)
                # right = np.zeros(n, dtype=ITYPE)

                for i in range(n):
                    point = node.points[i]

                    if data[point, split_dim] <  split_val:
                        left[idx_left] = point
                        idx_left += 1
                    else:
                        right[idx_right] = point
                        idx_right += 1

                if (idx_left > 0):
                    node.l = FairSplitTreeNode(left[0:idx_left].copy())
                    stack.append(node.l)

                if (idx_right > 0):
                    node.r = FairSplitTreeNode(right[0:idx_right].copy())
                    stack.append(node.r)
            else:
                node.center = data[node.points[0]]
                node.leaf = True


# @staticmethod
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef bint separated(FairSplitTreeNode node_a, FairSplitTreeNode node_b):
    return node_distances(node_a, node_b) >= max(node_a.diameter, node_b.diameter)


# @staticmethod
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef DTYPE_t node_distances(FairSplitTreeNode node_a, FairSplitTreeNode node_b):
    return euclidean(node_a.center, node_b.center) - node_a.diameter/2 - node_b.diameter/2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef DTYPE_t euclidean(np.ndarray[np.float64_t, ndim=1] v1, np.ndarray[np.float64_t, ndim=1] v2):
    cdef ITYPE_t i, m
    cdef DTYPE_t d = 0.0
    m = v1.shape[0]

    for i in xrange(m):
        d += (v1[i] - v2[i])**2

    return sqrt(d)


cdef class FairSplitTreeNode:

    cdef public np.ndarray points
    cdef public FairSplitTreeNode l, r
    cdef public np.float64_t diameter
    cdef public np.ndarray center
    cdef public bint leaf

    def __init__(
        self, 
        np.ndarray[np.int64_t, ndim=1] points, 
        FairSplitTreeNode left = None, 
        FairSplitTreeNode right = None, 
        np.ndarray[np.float64_t, ndim=1] center = None):
        
        self.points = points
        self.l = left
        self.r = right
        self.diameter = 0
        self.center = None
        self.leaf = False