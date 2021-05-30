import sys
import os
import time

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz, triu

from rng.rng import RelativeNeighborhoodGraph

from mst.mst import prim
from mst.mst import prim_plus
from mst.mst import prim_inc
from mst.mst import prim_graph
from mst.mst import prim_order


def _msts_rng(
    data,
    core_distances,
    knn,
    kmin = 1, kmax = 16, skip = 1, quick=True, efficient=True):

    start = time.time()

    # computes the RNG with regard to min_pts = kmax
    rng_object = RelativeNeighborhoodGraph(
        data, 
        core_distances, 
        knn, 
        kmax,
        efficient=efficient,
        quick=quick)
    
    # obtains the csr_matrix representation of the RNG
    rng = rng_object.graph()
    rng = rng.maximum(rng.T)
    end = time.time()
    
    print(end - start, end=' ')

    start = time.time()

    # loop over the values of mpts in the input range
    for i in range(kmin, kmax + 1, skip):
        
        # compute mst for mpts = i
        mst = prim_graph(
            rng.indices,
            rng.indptr,
            rng.data,
            np.ascontiguousarray(core_distances[:, i-1]),
            False)

    end = time.time()
    print(end - start, end=' ')
    print(int(rng.count_nonzero()/2), end=' ')


def _msts_coresg(
    data,
    core_distances,
    knn,
    kmin = 1, kmax = 16, skip = 1):
    
    # -----------------------------------
    start = time.time()
    
    # computes the MST w.r.t kmax and returns augmented kmax-NN information.
    mst, a_knn = prim_plus(
        data, 
        np.ascontiguousarray(core_distances[:, kmax-1]), 
        np.ascontiguousarray(knn[:, kmax-1]),
        False)

    # makes the mst an upper triangular matrix.
    mst = triu(mst.maximum(mst.T), format='csr')

    # augments the knng with the ties.
    knng = knng.maximum(a_knn.maximum(a_knn.T))
    
    # computes the CORE-SG graph w.r.t. the underlying distance. 
    nnsg = _coresg(mst, triu(knng))

    # eliminates zeroes from the matrix that might have remained from the operations.
    nnsg.eliminate_zeros()
    
    nnsg = nnsg.maximum(nnsg.T)

    end = time.time()
    print(end - start, end=' ')
    # -----------------------------------

    # -----------------------------------
    start = time.time()

    # loop over the values of mpts in the input range [kmin, kmax].
    for i in range(kmin, kmax, skip): 

        # compute mst for mpts = i
        mst = prim_graph(
            nnsg.indices,
            nnsg.indptr,
            nnsg.data,
            np.ascontiguousarray(core_distances[:, i-1]),
            False)

        # compute hierarchy for mpts = i
        # self._simplified_hierarchy(mst)

    end = time.time()
    print(end - start, end=' ')
    # -----------------------------------

    print(int(nnsg.count_nonzero()/2), end=' ')


def _msts_coresg_incremental(
    data, 
    core_distances,
    knn,
    knng,
    kmin = 1, kmax = 16, skip = 1):
    
    # -----------------------------------
    start = time.time()
    
    # computes the MST w.r.t kmax
    mst, a_knn = prim_plus(
        data, 
        np.ascontiguousarray(core_distances[:, kmax-1]), 
        np.ascontiguousarray(knn[:, kmax-1]),
        False)

    # makes the mst a symmetrix matrix.
    mst = mst.maximum(mst.T)

    # augments the knng with the ties.
    knng = knng.maximum(a_knn.maximum(a_knn.T))

    end = time.time()
    print(end - start, end=' ')
    # -----------------------------------

    # -----------------------------------
    start = time.time()

    # loop over the values of mpts in the input range [kmin, kmax].
    for i in range(kmax - 1, kmin - 1, -skip):

        # compute mst for mpts = i
        mst = prim_inc(
            data, 
            mst.indices, 
            mst.indptr,
            mst.data,
            knng.indices, 
            knng.indptr, 
            knng.data, 
            np.ascontiguousarray(core_distances[:, i-1]), 
            i, 
            False)

        # makes the resulting MST a symmetric graph for next iteration.
        mst = mst.maximum(mst.T)

        # eliminates the zero entries in the matrix (removing edges from the graph).
        knng.eliminate_zeros()


    end = time.time()
    print(end - start, end=' ')


def _knng(knn, core_distances, min_pts):
    n = len(knn)
    n_neighbors = min_pts - 1
    n_nonzero = n * n_neighbors

    knng = csr_matrix(
        (core_distances[:, 1:].ravel(), 
        knn[:, 1:].ravel(), 
        np.arange(0, n_nonzero + 1, n_neighbors)), 
        shape=(n, n))

    return knng.maximum(knng.T)


def _coresg(mst, knng, core_distances, min_pts):

    for current_point in range(mst.shape[0]):
        for i in range(mst.indptr[current_point], mst.indptr[current_point+1]):
            neighbor = mst.indices[i]
            if mst.data[i] == core_distances[current_point, min_pts-1] or \
                mst.data[i] == core_distances[neighbor, min_pts-1]:
                mst.data[i] = 0

    return mst.maximum(knng)
