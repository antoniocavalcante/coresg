import time

import numpy as np

from rng.rng import RelativeNeighborhoodGraph

from scipy.sparse.csgraph import minimum_spanning_tree

class HDBSCAN:

    def __init__(self, data_file, min_pts = 16, delimiter=',', distance='euclidean'):
        try:
            self.data = np.unique(np.genfromtxt(data_file, delimiter=delimiter), axis=0)
        except:
            print("Error reading the data file, please verify that the file exists.")            
        self.min_pts = min_pts
        self.distance = distance
        try:
            self.core_distances = np.genfromtxt(data_file + "-" + str(min_pts) + ".cd", delimiter=delimiter)
            self.knn = np.genfromtxt(data_file + "-" + str(min_pts) + ".knn", delimiter=delimiter)            
        except:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=min_pts, algorithm='ball_tree').fit(self.data)

            self.core_distances, self.knn = nbrs.kneighbors(self.data)
            # saving the computed core-distances and knn information on files.
            np.savetxt(data_file + "-" + str(self.min_pts) + ".cd" , self.core_distances, delimiter=delimiter)
            np.savetxt(data_file + "-" + str(self.min_pts) + ".knn", self.knn, delimiter=delimiter)

    def hdbscan(self, min_pts = 16):

        # computes minimum spanning tree (MST)

        return None


    def hdbscan(self, kmin = 1, kmax = 16, method='rng'):

        start = time.time()

        if method == 'mrg':
            g = None
        elif method == 'rng':
            g = RelativeNeighborhoodGraph(self.data)
        elif method == 'knn':
            g = None

        end = time.time()
        print(end - start)

        start = time.time()
        
        for i in range(kmin, kmax):
            # compute mst for mpts = i
            mst = minimum_spanning_tree(g.graph)
            # compute hierarchy for mpts = i
            self._construct_hierarchy()

        end = time.time()
        print(end - start)


        return None


    def _construct_hierarchy(self):

        from scipy.sparse.csgraph import depth_first_order 
        
        # get order of edges

        # index of max value

        # create level for that hierarchy

        # split array at that level and do the same in both halves

        return None


    def _construct_level(self):

        return None