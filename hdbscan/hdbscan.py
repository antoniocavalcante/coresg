import pyximport
pyximport.install()

import sys
import time

import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, save_npz, load_npz, triu

from rng.rng import RelativeNeighborhoodGraph

from mst.mst import prim
from mst.mst import prim_graph
from mst.mst import prim_order

class HDBSCAN:

    def __init__(self, datafile, min_pts = 16, delimiter=',', distance='euclidean'):

        sys.setrecursionlimit(10**6)
        
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

            nbrs = NearestNeighbors(n_neighbors=min_pts).fit(self.data)
            
            # knng_original = nbrs.kneighbors_graph(self.data, n_neighbors=min_pts, mode='distance')
            # print("[ORIGINAL] Number of edges: ", knng_original.count_nonzero())

            # computes the core-distances and knn information
            self.core_distances, self.knn = nbrs.kneighbors(self.data)

            self.knng = self._knng(self.min_pts)

            # saving the computed core-distances, knn and knng on files.
            np.savetxt(datafile + "-" + str(self.min_pts) + ".cd" , self.core_distances, delimiter=delimiter)
            np.savetxt(datafile + "-" + str(self.min_pts) + ".knn", self.knn, delimiter=delimiter, fmt='%i')
            save_npz(datafile + "-" + str(self.min_pts) + ".npz", self.knng)


    def hdbscan(self, min_pts = 16):

        # ------------------------------------
        # Time to compute the MST for kmax
        # ------------------------------------
        start = time.time()
        mst, _ = prim(self.data, self.core_distances[:, min_pts-1], False)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute hierarchy from MST
        # ------------------------------------
        start = time.time()
        hierarchy = self._simplified_hierarchy(mst)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute second MST
        # ------------------------------------
        start = time.time()
        mst, _ = prim(self.data, self.core_distances[:, int(min_pts/2)-1], False)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute second hierarchy
        # ------------------------------------
        start = time.time()
        hierarchy = self._simplified_hierarchy(mst)
        end = time.time()
        print(end - start, end=' ')

        return None


    def hdbscan_k(self, min_pts = 16):

        # ------------------------------------
        # Time to compute the MST for kmax
        # ------------------------------------
        start = time.time()
        mst, _ = prim(self.data, self.core_distances[:, min_pts-1], False)
        mst = triu(mst.maximum(mst.T), format='csr')
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to materialize the k-NNG
        # ------------------------------------
        start = time.time()
        self.knng = self._knng(self.min_pts)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to build the NNSG
        # ------------------------------------
        start = time.time()
        nnsg = self._nnsg(mst, self.knng)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute first MST
        # ------------------------------------
        start = time.time()
        mst = minimum_spanning_tree(nnsg)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute hierarchy from MST
        # ------------------------------------
        start = time.time()        
        # extracts simplified hierarchy from MST
        hierarchy = self._simplified_hierarchy(mst)
        end = time.time()
        print(end - start, end=' ')


        # print("PPPPPPPPPPPPPPP: ", int(min_pts/2))

        # ------------------------------------
        # Time to update the edges of the NNSG
        # ------------------------------------
        start = time.time()
        nnsg = self._update_edge_weights(nnsg, int(min_pts/2))
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute second MST
        # ------------------------------------
        start = time.time()
        mst = minimum_spanning_tree(nnsg)
        end = time.time()
        print(end - start, end=' ')

        # ------------------------------------
        # Time to compute second hierarchy
        # ------------------------------------
        start = time.time()        
        # extracts simplified hierarchy from MST
        hierarchy = self._simplified_hierarchy(mst)
        end = time.time()
        print(end - start, end=' ')

        # returns simplified hierarchy
        return None



    def _hdbscan_rng(self, kmin = 1, kmax = 16, quick=True):

        start = time.time()

        # computes the RNG with regard to min_pts = kmax
        rng_object = RelativeNeighborhoodGraph(self.data, self.core_distances, self.knn, kmax, quick=quick)
       
        # obtains the csr_matrix representation of the RNG
        rng = rng_object.graph()

        end = time.time()
        
        print(end - start, end=' ')

        start = time.time()

        time_msts = np.zeros((kmax - kmin + 1))

        # loop over the values of mpts in the input range
        for i in range(kmin, kmax + 1):
            # update base_graph wpythonith mutual reachability distances
            g = self._update_edge_weights(rng, i)
            
            start_mst = time.time()
            # compute mst for mpts = i
            mst = minimum_spanning_tree(g)

            time_msts[i - kmin] = time.time() - start_mst

            # compute hierarchy for mpts = i
            #self._construct_hierarchy(mst)

        end = time.time()
        print(end - start, end=' ')
        print(rng.count_nonzero(), end=' ')

        # write these results in a file
        # print(' '.join(map(str, time_msts)), end=' ')


    def _hdbscan_knn(self, kmin = 1, kmax = 16):
        
        # -----------------------------------
        start = time.time()
        
        # computes the MST w.r.t kmax
        mst, a_knn = prim(self.data, self.core_distances[:, kmax-1], False)
        # makes the mst an upper triangular matrix.
        mst = triu(mst.maximum(mst.T), format='csr')

        # augments the knng with the ties.
        self.knng.maximum(a_knn)

        # computes the nnsg graph w.r.t. the underlying distance. 
        nnsg = self._nnsg(mst, self.knng)
        # eliminates zeroes from the matrix that might have remained from the operations.
        nnsg.eliminate_zeros()

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------

        # -----------------------------------
        start = time.time()

        time_msts = np.zeros((kmax - kmin + 1))

        # loop over the values of mpts in the input range [kmin, kmax].
        for i in range(kmin, kmax):

            start_mst = time.time()

            nnsg = self._update_edge_weights(nnsg, i)

            # compute mst for mpts = i
            mst = minimum_spanning_tree(nnsg)

            time_msts[i - kmin] = time.time() - start_mst

            # compute hierarchy for mpts = i
            # self._simplified_hierarchy(mst)

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------

        print(nnsg.count_nonzero(), end=' ')

        # write these results in a file
        # print(' '.join(map(str, time_msts)), end=' ')


    def _hdbscan_knn_incremental(self, kmin = 1, kmax = 16):
        
        # -----------------------------------
        start = time.time()
        
        # computes the MST w.r.t kmax
        mst, _ = prim(self.data, self.core_distances[:, kmax-1], False)
        # makes the mst an upper triangular matrix.
        mst = mst.maximum(mst.T)
        # computes the nnsg graph w.r.t. the underlying distance. 
        nnsg = self._nnsg(mst, self.knng)
        # eliminates zeroes from the matrix that might have remained from the operations.
        nnsg.eliminate_zeros()

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------

        # -----------------------------------
        start = time.time()

        time_msts = np.zeros((kmax - kmin + 1))

        # loop over the values of mpts in the input range [kmin, kmax].
        for i in range(kmax - 1, kmin, -1):
            # print(i)

            start_mst = time.time()

            # compute mst for mpts = i
            mst = prim_graph(self.data, mst.tolil(), self.knn, self.core_distances, i, False)

            time_msts[i - kmin] = time.time() - start_mst

            # compute hierarchy for mpts = i
            self._simplified_hierarchy(mst)

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------


    def _update_edge_weights(self, nnsg, k):

        # retrieves the arrays with the indexes for rows and columns.
        row_ind, col_ind = nnsg.nonzero()

        # create matrix with core-distances corresponding to each (row, col) combination.
        nnsg.data = np.maximum(
            nnsg.data,
            self.core_distances[row_ind, k-1], 
            self.core_distances[col_ind, k-1])
        
        # returns the nnsg with updated edge weights.
        return nnsg


    def _update_graph(self, mst, k):

        # compute knn graph for this value of mpts.
        knng = self._knng(k)

        # computes the NNSG
        nnsg = self._nnsg(mst, knng)
        
        # rows and columns indices of nonzero positions.
        row_ind, col_ind = nnsg.nonzero()

        # update graph with mutual reachability distance
        nnsg.data = np.maximum(
            nnsg.data,
            self.core_distances[row_ind, k-1], 
            self.core_distances[col_ind, k-1])
        
        # return maximum between knn and
        return nnsg


    def _knng(self, min_pts):
        n_neighbors = min_pts - 1
        n_nonzero = self.n * n_neighbors

        knng = csr_matrix(
            (self.core_distances[:, 1:].ravel(), 
            self.knn[:, 1:].ravel(), 
            np.arange(0, n_nonzero + 1, n_neighbors)), 
            shape=(self.n, self.n))

        # print("Number of edges: ", knng.count_nonzero())

        knng = triu(knng.maximum(knng.T))

        return knng


    def _nnsg(self, mst, knng):

        # converts MST to LIL format.
        mst = mst.tolil()

        # nonzero positions of knng.
        row_ind, col_ind = knng.nonzero()

        # sets all positions to zero so we can return the sum/max of both matrices.
        mst[row_ind, col_ind] = 0

        # returns sum MST and KNNG in CSR format.
        return mst.tocsr().maximum(knng)


    def _construct_hierarchy(self):

        from scipy.sparse.csgraph import depth_first_order 
        
        # get order of edges

        # index of max value

        # create level for that hierarchy

        # split array at that level and do the same in both halves

        return None


    def _simplified_hierarchy(self, mst):
        
        mst = mst.maximum(mst.T)

        nodes, reachability = prim_order(mst.data, mst.indices, mst.indptr, self.n)

        # hierarchy = self._get_nodes(reachability, 0, self.n)

        return None


    def _get_nodes(self, reachability, start, end):

        if end - start < 2:
            return None

        split = start + 1 + np.argmax(reachability[start+1:end])
        # print("-----------------------------------")
        # print(reachability[start+1:end])
        # print(start+1, end, split)

        d = {}

        d['level'] = reachability[split]
        d['start'] = start
        d['end']   = end
        
        d['children'] = [self._get_nodes(reachability, start, split),
        self._get_nodes(reachability, split + 1, end)]

        return d