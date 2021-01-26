import numpy as np

from rng.rng import RelativeNeighborhoodGraph

from scipy.sparse.csgraph import minimum_spanning_tree

class HDBSCAN:

    def __init__(self, data_file, delimiter=',', distance='euclidean'):
        self.data = np.genfromtxt(data_file, delimiter=delimiter)
        self.min_pts = min_pts
        self.distance = distance

    def hdbscan(self):

        # read data, core-distances and k-nn

        # computes minimum spanning tree (MST)

        # builds clustering hierarchy from MST


        return None


    def hdbscan(self, kmin = 1, kmax = 16, method='rng'):


        if method == 'mrg':
            g = None
        elif method == 'rng':
            g = RelativeNeighborhoodGraph(self.data)
        elif method == 'knn':
            g = None

        for i in range(kmin, kmax):
            # compute mst for mpts = i
            mst = minimum_spanning_tree(g)

            # compute hierarchy for mpts = i
            
        return None

    def _read_data(self):

        return None


    def _read_knn(self):

        return None


    def _read_core_distances(self):

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