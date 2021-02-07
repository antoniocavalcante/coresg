import numpy as np

import mst

import pyximport; pyximport.install()

from sklearn.neighbors import NearestNeighbors

from rng.fair_split_tree import FairSplitTree

if __name__ == "__main__":
    
    data = np.random.rand(1600,16)

    k = 16

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)

    # computes the core-distances and knn information
    core_distances, knn = nbrs.kneighbors(data)

    fst = FairSplitTree(data, core_distances[:, k-1])