import pstats, cProfile

import pyximport
pyximport.install()

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz, triu

from mst.mst import prim

from sklearn.neighbors import NearestNeighbors

data = np.unique(
    np.genfromtxt(
        "../dataset-performance/16d-16.dat", 
        delimiter=" "), 
        axis=0)

nbrs = NearestNeighbors(n_neighbors=16).fit(data)

core_distances, knn = nbrs.kneighbors(data)

cProfile.runctx("prim(data, np.ascontiguousarray(core_distances[:, 15]), False)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()