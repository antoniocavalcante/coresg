import numpy as np
cimport numpy as np

cimport cython
import time
from scipy.spatial import distance
from scipy.sparse import csr_matrix

from rng.fair_split_tree import FairSplitTree, FairSplitTreeNode

include '../mst/parameters.pxi'

cdef class RelativeNeighborhoodGraph:

    cdef np.ndarray data, core_distances, knn
    cdef bint quick, naive

    cdef ITYPE_t min_points

    cdef list u, v, w

    def __init__(self, 
        np.ndarray data, 
        np.ndarray core_distances,
        np.ndarray knn,
        ITYPE_t min_points, bint quick = True, bint naive = False):
        
        self.data = data

        self.core_distances = core_distances[:, min_points - 1]
        self.knn = knn

        self.min_points = min_points

        self.quick = quick
        self.naive = naive

        self.u = []
        self.v = []
        self.w = []

        # start = time.time()
        # Build Fair Split Tree
        T = FairSplitTree(self.data, self.core_distances)
        # end = time.time()

        # print("[FST] " + str(end - start))

        # start = time.time()
        # Find Well-Separated Pairs and their respective SBCN
        self.wspd(T)
        # end = time.time()

        # print("[WSPD] " + str(end - start))


    cpdef graph(self):
        cdef ITYPE_t n = len(self.data)
        
        return csr_matrix((self.w, (self.u, self.v)), shape=(n, n))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void sbcn(self, ITYPE_t[:] red, ITYPE_t[:] blue):

        cdef ITYPE_t r, b, r_size, b_size

        r_size = red.size
        b_size = blue.size

        # if both sets are singletons
        if r_size == 1 and b_size == 1:
            self.add_edge(
                red[0],
                blue[0],
                distance.euclidean(self.data[red[0]], self.data[blue[0]]))
            return
        
        cdef DTYPE_t min_dist

        cdef DTYPE_t[:] distances

        cdef DTYPE_t[:] min_distances_rb = np.full(red.size,  np.inf, dtype=DTYPE)
        cdef DTYPE_t[:] min_distances_br = np.full(blue.size, np.inf, dtype=DTYPE)

        cdef ITYPE_t[:] min_distances_points_rb = np.zeros(red.size,  dtype=ITYPE)
        cdef ITYPE_t[:] min_distances_points_br = np.zeros(blue.size, dtype=ITYPE)

        for r in range(r_size):
            
            distances = distance.cdist([self.data[red[r]]], self.data[blue])[0]
            
            for i in range(b_size):
                
                d_rb = max(distances[i], self.core_distances[red[r]], self.core_distances[blue[i]])
                
                if d_rb < min_distances_rb[r]:
                    min_distances_rb[r] = d_rb
                    min_distances_points_rb[r] = i
            
            b = min_distances_points_rb[r]

            if min_distances_rb[r] < min_distances_br[b]:
                min_distances_br[b] = min_distances_rb[r]
                min_distances_points_br[b] = r
        
        for r in range(r_size):
            b = min_distances_points_rb[r]
            if min_distances_points_br[b] == r:
                self.add_edge(red[r], blue[b], 1)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void add_edge(self, ITYPE_t point_a, ITYPE_t point_b, DTYPE_t weight):
        if self.relative_neighbors(point_a, point_b):
            self.u.append(point_a)
            self.v.append(point_b)
            self.w.append(1)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def wspd(self, fst):
        stack = [fst.root]

        while stack:
            node = stack.pop()

            if not node.l.leaf:
                stack.append(node.l)

            if not node.r.leaf:
                stack.append(node.r) 

            self.find_pairs(node.l, node.r)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void find_pairs(self, node_a, node_b):

        if FairSplitTree.separated(node_a, node_b):
            self.sbcn(node_a.points, node_b.points)
        else:
            if node_a.diameter <= node_b.diameter:
                self.find_pairs(node_a, node_b.l)
                self.find_pairs(node_a, node_b.r)
            else:
                self.find_pairs(node_a.l, node_b)
                self.find_pairs(node_a.r, node_b)
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint relative_neighbors(self, ITYPE_t point_a, ITYPE_t point_b):

        if self.quick:
            if not self._relative_neighbors_quick(point_a, point_b):
                return False
            
        if self.naive:
            if not self._relative_neighbors_naive(point_a, point_b):
                return False

        return True


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint _relative_neighbors_quick(self, ITYPE_t point_a, ITYPE_t point_b):
        return True


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint _relative_neighbors_naive(self, ITYPE_t point_a, ITYPE_t point_b):

        distance_ab = distance.euclidean(self.data[point_a], self.data[point_b])

        for point_c in range(len(self.data)):
            if distance_ab > max(
                distance.euclidean(self.data[point_a], self.data[point_c]), 
                distance.euclidean(self.data[point_b], self.data[point_c])):
                return False

        return True


    # cpdef naive_constructor(self):

    #     ITYPE_t i, j, n

    #     n = len(data)

    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             dij = distance.euclidean(data[i], data[j])

    #             rn = True

    #             for m in range(n):
    #                 dim = distance.euclidean(data[i], data[m])
    #                 djm = distance.euclidean(data[j], data[m])
    #                 if (dij > max(dim, djm)):
    #                     rn = False
    #                     break
                
    #             if rn:
    #                 u.append(i)
    #                 v.append(j)
    #                 w.append(dij)

    #     return graph = csr_matrix((w, (u, v)), shape=(n, n))


if __name__ == "__main__":
    
    # generates a small random dataset
    data = np.array([
        [0, 2],
        [1, 1], 
        [1, 2], 
        [1, 3], 
        [2, 2], 
        [3, 1], 
        [4, 5],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 7],
        [7, 1],
        [7, 2],
        [8, 1],
        [8, 2]])
    
    rng = RelativeNeighborhoodGraph(data, quick=False, naive=True)

    n = len(data)

    print(rng.graph)

    print("------------------------------------")

    u = []
    v = []
    w = []

    for i in range(n):
        for j in range(i + 1, n):
            dij = distance.euclidean(data[i], data[j])

            rn = True

            for m in range(n):
                dim = distance.euclidean(data[i], data[m])
                djm = distance.euclidean(data[j], data[m])
                if (dij > max(dim, djm)):
                    rn = False
                    break
            
            if rn:
                u.append(i)
                v.append(j)
                w.append(dij)

    graph = csr_matrix((w, (u, v)), shape=(n, n))
    print(graph)
