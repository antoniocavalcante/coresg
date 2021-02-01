import pyximport; 
pyximport.install()

import time

import numpy as np

from sklearn.neighbors import NearestNeighbors

from mst import mst

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from rng.rng import RelativeNeighborhoodGraph

import pstats, cProfile


if __name__ == "__main__":
    
    # value of k for the k-NN
    k = 4

    # generates a small random dataset
    data = np.random.rand(10,2)
    
    # computes core-distances and knn information
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)    
    core_distances, knn = nbrs.kneighbors(data)

    # computes MST based on the local implementation of Prim
    start = time.time()
    mst_prim = mst.prim(data, core_distances, k, False)
    end = time.time()
    print("[PRIM] " + str(end - start))
    print(" ---------------------- ")

    # # computes MST based on the local implementation of Prim
    # start = time.time()
    # mst_split = mst.split_mst(data, core_distances, knn, k, False)
    # end = time.time()
    # print("[SPLT] " + str(end - start))
    # print(" ---------------------- ")

    # # computes MST based on the scikit-learn package
    # X = squareform(pdist(data, 'euclidean'))
    # start = time.time()
    # mst_scip = minimum_spanning_tree(X)
    # end = time.time()
    # print("[BLTIN] " + str(end - start))
    # print(" ---------------------- ")


    start = time.time()
    rng = RelativeNeighborhoodGraph(data, core_distances, knn, k)
    end = time.time()
    print("[RNG] " + str(end - start))
    print(" ---------------------- ")

    graph = rng.graph()

    start = time.time()
    mst_scip = minimum_spanning_tree(graph)
    end = time.time()

    # print(rng.graph())

    print(mst_prim.sum(), mst_scip.sum())
    # # compares the total weight of both graphs
    # if mst_prim.sum() == mst_split.sum():
    #     print("Exactly the same weight!")
    # else:
    #     print("Maybe there is something wrong!")
