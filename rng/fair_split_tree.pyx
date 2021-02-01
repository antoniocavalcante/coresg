# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as np

cimport cython

from scipy.spatial import distance

include '../mst/parameters.pxi'

cdef class FairSplitTree:

    cdef public FairSplitTreeNode root

    def __init__(self, np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] core_distances):
 
        self.root = FairSplitTreeNode(np.arange(len(data), dtype=ITYPE))

        self.construct(data, core_distances)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef construct(self, np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] core_distances):

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

            n = node.points.size

            if n > 1:
                # finds the maximum and minimum values for each dimension
                maxdim = np.amax(data[node.points], axis=0)
                mindim = np.amin(data[node.points], axis=0)

                # updates the diameter of the node (euclidean distance)
                node.diameter = max(
                    distance.euclidean(maxdim, mindim),
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


    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def separated(FairSplitTreeNode node_a, FairSplitTreeNode node_b):
        return FairSplitTree.node_distances(node_a, node_b) > max(node_a.diameter, node_b.diameter)


    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def node_distances(FairSplitTreeNode node_a, FairSplitTreeNode node_b):
        return distance.euclidean(node_a.center, node_b.center) - node_a.diameter/2 - node_b.diameter/2


cdef class FairSplitTreeNode:

    cdef readonly ITYPE_t[:] points
    cdef public FairSplitTreeNode l, r
    cdef public DTYPE_t diameter
    cdef public DTYPE_t[:] center
    cdef public bint leaf

    def __init__(self, ITYPE_t[:] points, FairSplitTreeNode left = None, FairSplitTreeNode right = None, DTYPE_t[:] center = None):
        self.points = points
        self.l = left
        self.r = right
        self.diameter = 0
        self.center = None
        self.leaf = False    