import numpy as np

from sklearn.neighbors import NearestNeighbors

from mst import mst

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

if __name__ == "__main__":
    
    # generates a 
    k = 2

    # generates a small random dataset
    data = np.random.rand(10,2)
    
    # computes core-distances and knn information
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)    
    core_distances, knn = nbrs.kneighbors(data)

    # computes MST based on the local implementation of Prim
    mst_prim = mst.prim(data, core_distances, k, False)
    print(mst_prim)
    print(mst_prim.sum())
    X = squareform(pdist(data, 'euclidean'))

    print(" ---------------------- ")

    # computes MST based on the scikit-learn package
    mst_scip = minimum_spanning_tree(X)
    print(mst_scip)
    print(mst_prim.sum())

    # compares the total weight of both graphs
    if mst_prim.sum() == mst_scip.sum():
        print("Pass!")
    else:
        print("Something is wrong!")
