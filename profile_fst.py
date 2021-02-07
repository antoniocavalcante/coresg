import pstats, cProfile

import time

import numpy as np

from sklearn.neighbors import NearestNeighbors

import pyximport; pyximport.install()

from rng.rng import RelativeNeighborhoodGraph

from mst.mst import prim

# data = np.random.rand(1000,16)

data = np.unique(np.genfromtxt('/home/toni/git/dataset-performance/16d-5.dat', delimiter=' '), axis=0)

nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(data)

# computes the core-distances and knn information
core_distances, knn = nbrs.kneighbors(data)

knn = np.array(knn, dtype=np.int64)

# start = time.time()
# rng = RelativeNeighborhoodGraph(data, core_distances, knn, 16)
# g = rng.graph()

# end = time.time()

# print("[RNG] " + str(end - start))
# print("[#edges] " + str(g.count_nonzero()))

# start = time.time()
# mst_kmax = prim(data, core_distances, 16, False)
# g = rng.graph()
# end = time.time()

# print("[MST] " + str(end - start))
# print("[#edges] " + str(mst_kmax.count_nonzero()))


cProfile.runctx("RelativeNeighborhoodGraph(data, core_distances, knn, 16, quick=False)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
