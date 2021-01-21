import numpy as np

from sklearn.neighbors import NearestNeighbors

from mst import mst

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

if __name__ == "__main__":
    
    # value of k for the k-NN
    k = 1

    # generates a small random dataset
    data = np.random.rand(10,2)
    
    # computes core-distances and knn information
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)    
    core_distances, knn = nbrs.kneighbors(data)

    # computes MST based on the local implementation of Prim
    mst_prim = mst.prim(data, core_distances, k, False)
    print(mst_prim)
    print(mst_prim.sum())

    print(" ---------------------- ")

    # computes MST based on the scikit-learn package
    X = squareform(pdist(data, 'euclidean'))
    mst_scip = minimum_spanning_tree(X)
    print(mst_scip)
    print(mst_scip.sum())

    # compares the total weight of both graphs
    if mst_prim.sum() == mst_scip.sum():
        print("Exactly the same weight!")
    else:
        print("Maybe there is something wrong!")
