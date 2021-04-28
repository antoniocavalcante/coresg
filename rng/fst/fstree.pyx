import math

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport sqrt

include '../../parameters.pxi'

cdef class FairSplitTree:

    cdef FairSplitTreeNode root

    def __init__(
        self, 
        np.ndarray[DTYPE_t, ndim=2] data, 
        np.ndarray[DTYPE_t, ndim=1] core_distances):
 
        self.root = FairSplitTreeNode(np.arange(data.shape[0], dtype=ITYPE))

        self.construct(data, core_distances)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void construct(
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

        left  = np.zeros(data.shape[0], dtype=ITYPE)
        right = np.zeros(data.shape[0], dtype=ITYPE)

        while stack:
            
            node = stack.pop()

            n = node.points.shape[0]

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


cdef class FairSplitTreeNode:

    cdef np.ndarray points
    cdef FairSplitTreeNode l, r
    cdef np.float64_t diameter
    cdef np.ndarray center
    cdef bint leaf

    def __init__(
        self, 
        np.ndarray[np.int64_t, ndim=1] points):
        
        self.points = points
        self.l = None
        self.r = None
        self.diameter = 0
        self.center = None
        self.leaf = False


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