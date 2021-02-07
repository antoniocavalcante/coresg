# cython: profile=True
from __future__ import print_function

import sys
import time

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport sqrt

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from rng.fair_split_tree import FairSplitTree, FairSplitTreeNode, separated

include '../parameters.pxi'

cdef class RelativeNeighborhoodGraph:

    cdef bint quick, naive
    cdef ITYPE_t min_points, n
    cdef list u, v, w

    def __init__(self, 
        np.ndarray[DTYPE_t, ndim=2] data,
        np.ndarray[DTYPE_t, ndim=2] core_distances,
        np.ndarray[ITYPE_t, ndim=2] knn,
        ITYPE_t min_points, 
        bint quick = True, 
        bint naive = False):
        
        self.min_points = min_points

        self.quick = quick
        self.naive = naive

        self.u = []
        self.v = []
        self.w = []

        self.n = data.shape[0]

        # start = time.time()
        # Build Fair Split Tree
        T = FairSplitTree(data, core_distances[:, min_points - 1])
        # end = time.time()

        # print("[FST]: " + str(end - start))

        # start = time.time()
        # Find Well-Separated Pairs and their respective SBCN
        self.wspd(T, data, core_distances[:, min_points - 1], knn)
        # end = time.time()

        # print("[WSPD]: " + str(end - start))


    cpdef graph(self):
        return csr_matrix((self.w, (self.u, self.v)), shape=(self.n, self.n))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void wspd(
        self, 
        fst,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):
        
        cdef list stack = [fst.root]

        while stack:
            node = stack.pop()

            if not node.l.leaf:
                stack.append(node.l)

            if not node.r.leaf:
                stack.append(node.r) 

            self.find_pairs(node.l, node.r, data, core_distances, knn)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void find_pairs(
        self, 
        node_a,
        node_b,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):

        cdef list stack = [(node_a, node_b)]

        while stack:
            current_a, current_b = stack.pop()

            if separated(current_a, current_b):
                self.sbcn(current_a.points, current_b.points, data, core_distances, knn)
            else:
                if current_a.diameter <= current_b.diameter:
                    stack.append((current_a, current_b.l))
                    stack.append((current_a, current_b.r))
                else:
                    stack.append((current_a.l, current_b))
                    stack.append((current_a.r, current_b))


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # @cython.initializedcheck(False)
    # cdef void find_pairs(
    #     self, 
    #     node_a,
    #     node_b,
    #     np.ndarray[DTYPE_t, ndim=2] data,
    #     np.ndarray[DTYPE_t, ndim=1] core_distances,
    #     np.ndarray[ITYPE_t, ndim=2] knn):

    #     if separated(node_a, node_b):
    #         self.sbcn(node_a.points, node_b.points, data, core_distances, knn)
    #     else:
    #         if node_a.diameter <= node_b.diameter:
    #             self.find_pairs(node_a, node_b.l, data, core_distances, knn)
    #             self.find_pairs(node_a, node_b.r, data, core_distances, knn)
    #         else:
    #             self.find_pairs(node_a.l, node_b, data, core_distances, knn)
    #             self.find_pairs(node_a.r, node_b, data, core_distances, knn)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void sbcn(self, 
        ITYPE_t[:] red, 
        ITYPE_t[:] blue,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):

        cdef ITYPE_t r, b, r_size, b_size, point_r, point_b, vertex_1, vertex_2
        cdef DTYPE_t min_dist, d_rb

        r_size = red.shape[0]
        b_size = blue.shape[0]

        min_dist = float("inf")

        point_r = 0
        point_b = 0

        vertex_1 = -1
        vertex_2 = -1

        for r in xrange(r_size):
            point_r = red[r]

            for b in xrange(b_size):
                point_b = blue[b]

                d_rb = _mutual_reachability_distance(point_r, point_b, data, core_distances)

                if d_rb < min_dist:
                    vertex_1 = point_r
                    vertex_2 = point_b
                    min_dist = d_rb


        # add edge
        self.add_edge(vertex_1, vertex_2, min_dist, data, core_distances, knn)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef void add_edge(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight, 
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):
        
        if self.relative_neighbors(point_a, point_b, weight, data, core_distances, knn):
            self.u.append(point_a)
            self.v.append(point_b)
            self.w.append(weight)
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint relative_neighbors(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):

        if self.quick:
            if not self._relative_neighbors_quick(point_a, point_b, weight, data, core_distances, knn):
                return False
            
        if self.naive:
            if not self._relative_neighbors_naive(point_a, point_b, weight, data, core_distances):
                return False

        return True


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint _relative_neighbors_quick(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances,
        ITYPE_t[:, :] knn):

        cdef ITYPE_t c, point_c

        if weight == max(core_distances[point_a], core_distances[point_b]):
            return True

        for c in xrange(1, self.min_points):
            point_c = knn[point_a, c]

            if core_distances[point_c] < weight:
                if weight > max(
                    _mutual_reachability_distance(point_a, point_c, data, core_distances), 
                    _mutual_reachability_distance(point_b, point_c, data, core_distances)):
                    return False

        for c in xrange(1, self.min_points):
            point_c = knn[point_b, c]

            if core_distances[point_c] < weight:
                if weight > max(
                    _mutual_reachability_distance(point_a, point_c, data, core_distances), 
                    _mutual_reachability_distance(point_b, point_c, data, core_distances)):
                    return False

        return True


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cdef bint _relative_neighbors_naive(
        self, 
        ITYPE_t point_a, 
        ITYPE_t point_b, 
        DTYPE_t weight,
        DTYPE_t[:, :] data,
        DTYPE_t[:] core_distances):

        cdef ITYPE_t point_c

        for point_c in xrange(self.n):
            if weight > max(
                _mutual_reachability_distance(point_a, point_c, data, core_distances), 
                _mutual_reachability_distance(point_b, point_c, data, core_distances)):
                return False

        return True


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef DTYPE_t _mutual_reachability_distance(
    ITYPE_t point_a, 
    ITYPE_t point_b,
    DTYPE_t[:, :] data,
    DTYPE_t[:] core_distances):

    return max(
        euclidean_local(data[point_a], data[point_b]),
        core_distances[point_a],
        core_distances[point_b])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef DTYPE_t euclidean_local(DTYPE_t[:] v1, DTYPE_t[:] v2):
    cdef ITYPE_t i, m
    cdef DTYPE_t d = 0.0
    m = v1.shape[0]
 
    for i in xrange(m):
        d += (v1[i] - v2[i])**2

    return sqrt(d)


# cdef DTYPE_t euclidean_local(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2):
