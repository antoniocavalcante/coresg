# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as np

cimport cython

from scipy.spatial import distance

include '../mst/parameters.pxi'

cdef class FairSplitTree:

    cdef public FairSplitTreeNode root

    def __init__(self, np.ndarray[DTYPE_t, ndim=2] data):
 
        self.root = FairSplitTreeNode(np.arange(len(data), dtype=ITYPE))

        self.construct(data)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def construct(self, np.ndarray[DTYPE_t, ndim=2] data = np.random.rand(10000,2)):

        cdef FairSplitTreeNode node
        
        cdef ITYPE_t point, split_dim
        cdef ITYPE_t[:] left, right
        cdef DTYPE_t split_val
        cdef np.ndarray[DTYPE_t, ndim=1] maxdim, mindim

        # starts with the root of the tree
        cdef list stack = [self.root]

        while stack:
            
            node = stack.pop()

            if len(node.points) > 1:
                maxdim = np.amax(data[node.points], axis=0)
                mindim = np.amin(data[node.points], axis=0)

                # updates the diameter of the node (euclidean distance)
                node.diameter = distance.euclidean(maxdim, mindim)
                
                # updates the geometric center of this node
                node.center = (maxdim + mindim)/2

                split_dim = np.argmax(maxdim - mindim)
                split_val = (mindim[split_dim] + maxdim[split_dim]) / 2
                
                left  = np.array([point for point in node.points if data[point, split_dim] <  split_val], dtype=ITYPE)
                right = np.array([point for point in node.points if data[point, split_dim] >= split_val], dtype=ITYPE)

                if (left.size > 0):
                    node.l = FairSplitTreeNode(left)
                    stack.append(node.l)

                if (right.size > 0):
                    node.r = FairSplitTreeNode(right)
                    stack.append(node.r)
            else:
                node.center = data[node.points[0]]
                node.leaf = True


    @staticmethod
    def separated(node_a, node_b):
        return FairSplitTree.node_distances(node_a, node_b) > max(node_a.diameter, node_b.diameter)


    @staticmethod
    def node_distances(node_a, node_b):
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