from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz, triu
from sklearn.neighbors import NearestNeighbors


import numpy as np

# min_pts = 10

# k = 9

# # data = np.genfromtxt("/home/toni/git/dataset-performance/16d-1.dat", delimiter=' ')
# data = np.genfromtxt("/home/toni/Dropbox/UofA/Research/HDBSCAN'/data/jad/jad.data", delimiter=',')

# n = data.shape[0]

# nbrs = NearestNeighbors(n_neighbors=min_pts, algorithm='ball_tree').fit(data)

# # computes the core-distances and knn information
# core_distances, knn = nbrs.kneighbors(data)

# knng = csr_matrix(
#     (core_distances[:, 1:min_pts].ravel(), 
#     knn[:, 1:min_pts].ravel(), 
#     np.arange(0, (n * (min_pts-1)) + 1, min_pts-1)), 
#     shape=(n, n))

# knng.eliminate_zeros()

# print(knng.count_nonzero())


# knng = triu(knng.maximum(knng.T), format='csr')


# knng_scipy = nbrs.kneighbors_graph(n_neighbors=k)

# knng_scipy.eliminate_zeros()

# print(knng_scipy.count_nonzero())


def change_graph(graph):
    graph[0, 1] = 10


g = lil_matrix((5, 5))

print(g.toarray())

change_graph(g)

print(g.toarray())

