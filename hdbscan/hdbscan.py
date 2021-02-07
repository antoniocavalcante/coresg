import pyximport
pyximport.install()

import time

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, save_npz, load_npz, triu

from rng.rng import RelativeNeighborhoodGraph

from mst.mst import prim

class HDBSCAN:

    def __init__(self, datafile, min_pts = 16, delimiter=',', distance='euclidean'):
        try:
            self.data = np.unique(np.genfromtxt(datafile, delimiter=delimiter), axis=0)
        except:
            print("Error reading the data file, please verify that the file exists.")        

        self.n = len(self.data)
        self.min_pts = min_pts
        self.distance = distance

        try:
            self.core_distances = np.genfromtxt(datafile + "-" + str(min_pts) + ".cd", delimiter=delimiter)
            self.knn = np.genfromtxt(datafile + "-" + str(min_pts) + ".knn", delimiter=delimiter, dtype=np.int64)
            self.knng = load_npz(datafile + "-" + str(self.min_pts) + ".npz")
        except:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=min_pts, algorithm='ball_tree').fit(self.data)

            # computes the core-distances and knn information
            self.core_distances, self.knn = nbrs.kneighbors(self.data)

            # computes the directed k-nearest neighbors graph
            directed_knng = nbrs.kneighbors_graph(mode='distance')

            # stores an upper triangular version of the k-nn graph
            self.knng = triu(directed_knng.maximum(directed_knng.T), format='csr')
            self.knng.eliminate_zeros()

            # saving the computed core-distances, knn and knng on files.
            np.savetxt(datafile + "-" + str(self.min_pts) + ".cd" , self.core_distances, delimiter=delimiter)
            np.savetxt(datafile + "-" + str(self.min_pts) + ".knn", self.knn, delimiter=delimiter, fmt='%i')
            save_npz(datafile + "-" + str(self.min_pts) + ".npz", self.knng)


    def hdbscan(self, min_pts = 16):

        # computes minimum spanning tree (MST)
        
        # computes hierarchy for min_pts

        return None


    def hdbscan_g(self, kmin = 1, kmax = 16, method='knn', quick=True):

        start = time.time()

        if method == 'rng':
            # computes the RNG with regard to min_pts = kmax
            rng_object = RelativeNeighborhoodGraph(self.data, self.core_distances, self.knn, kmax, quick=quick)
            # obtains the csr_matrix representation of the RNG
            rng = rng_object.graph()
            # makes the RNG an upper triangular matrix
            rng = triu(rng.maximum(rng.T))

            base_graph = self._graph_setup(rng)

        elif method == 'knn':
            # computes the MST_kmax + kmax-NN graph
            mst_kmax = prim(self.data, self.core_distances, kmax, False)
            # makes the mst an upper triangular matrix
            mst_kmax = triu(mst_kmax.maximum(mst_kmax.T), format='csr')

            base_graph = self._graph_setup(mst_kmax)
        
        # # eliminates zeroes from the matrix that might have remained from the operations.
        base_graph.eliminate_zeros()

        end = time.time()
        
        print(end - start, end=' ')

        start = time.time()

        # loop over the values of mpts in the input range
        for i in range(kmin, kmax):
            # update base_graph with mutual reachabilit
            g = self._update_graph(base_graph, i)

            # compute mst for mpts = i
            mst = minimum_spanning_tree(g)

            # compute hierarchy for mpts = i
            #self._construct_hierarchy(mst)

        end = time.time()
        print(end - start, end=' ')
        print(base_graph.count_nonzero(), end=' ')


    def _update_graph(self, graph, min_pts):
        # retrieves the arrays with the indexes for rows and columns.
        row_ind, col_ind = graph.nonzero()

        # create matrix with core-distances corresponding to each (row, col) combination.
        w = np.maximum(self.core_distances[row_ind, min_pts-1], self.core_distances[col_ind, min_pts-1])
        
        # returns a matrix of mutual reachability distances.
        return csr_matrix((w, (row_ind, col_ind)), shape=(self.n, self.n)).maximum(graph)


    def _graph_setup(self, graph):
        graph = graph.todok()

        # create difference graph from knn
        row_ind, col_ind = self.knng.nonzero()

        # sets 
        for row, col in zip(row_ind, col_ind):
            graph[row, col] = 0

        # return sum of difference and mst_kmax
        return graph.tocsr() + self.knng


    def _construct_hierarchy(self):

        from scipy.sparse.csgraph import depth_first_order 
        
        # get order of edges

        # index of max value

        # create level for that hierarchy

        # split array at that level and do the same in both halves

        return None


    def _construct_level(self):

        return None