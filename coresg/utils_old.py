import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from patches import circles
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
from heapq import heapify, heappop, heappush, heappushpop, merge
from scipy.spatial.distance import minkowski
import random
from itertools import groupby
import hdbscan
from scipy.cluster.hierarchy import linkage, dendrogram
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list, to_tree, leaders
import matplotlib.cm as cm
import matplotlib.colors as colors
import resources.hai as hai
# from hierarchy_tree import HierarchyTree


def convert_pynndescent_output(neighbors):

    n = len(neighbors[0])
    k = len(neighbors[0][0])-1
    
    knn = [[(neighbors[1][i][j+1], neighbors[0][i][j+1]) for j in range(k)] for i in range(n)]
    
    return knn

def get_sorted_dist_matrix(dataset):
    """
    Description:
        Computes the distance matrix and sorts it.
    
    Parameters:
        dataset (2d Array): [[row1],[row2],...] dataset where each row is an object.

    Returns:
        (2d Array of tuples (dist, index)): returns a 2d Array of tuples (dist, index) representing the distance to the
        "index" object. This Array contains all distances object-to-object already sorted ascendingly.
        
        Note: the distance of an object to itself is not include.
    """
    dist_m = distance_matrix(dataset, dataset)
    dist_m = [[(dist, i) for i, dist in enumerate(a)] for a in dist_m]
    sorted_dist_m = [sorted(a) for a in dist_m]
    sorted_dist_m = [a[1:] for a in sorted_dist_m]
    return sorted_dist_m


def get_rbg(sorted_dist_matrix, min_k=1):
    """
    Description:
        Computes the RBG given a sorted distance matrix and a min K.
    
    Parameters:
        sorted_dist_matrix (2d Array of tuples (dist, index)): 2d Array with all distances object-to-object already sorted.
        min_k (int >= 1): especifies the number of neighbors to be considered to compute the uniform limit.
   
    Returns:
        (2d Array of tuples (dist, index)): returns the RBG - a 2d Array like sorted_dist_matrix, but without the edges bigger
        than the uniform limit found.
    """
    assert min_k > 0
    
    def binarySearch(arr, l, r, x):
        while r > l:
            mid = l + (r - l) // 2
            if arr[mid][0] == x:
                while mid < len(arr) and arr[mid][0] <= x:
                    mid += 1
                return mid
            elif arr[mid][0] > x:
                r = mid - 1 
            else: 
                l = mid + 1 
        while r < len(arr) and arr[r][0] <= limit:
            r += 1
        return r
    
    def sliceUniform(neighbors, limit):
        i = binarySearch(neighbors, 0, len(neighbors)-1, limit)
        return neighbors[:i]
    
    limit = max([a[min_k-1][0] for a in sorted_dist_matrix])
    return limit, [sliceUniform(a, limit) for a in sorted_dist_matrix]


def get_knn(sorted_dist_matrix, k=1):
    """
    Description:
        Computes the KNN given a sorted distance matrix and a K.
    
    Parameters:
        sorted_dist_matrix (2d Array of tuples (dist, index)): 2d Array with all distances object-to-object already sorted.
        k (int >= 1): especifies the number of neighbors.
    
    Returns:
        (2d Array of tuples (dist, index)): returns the KNN - a 2d Array like sorted_dist_matrix, but with only k edges in
        each row.
    """
    assert k > 0
    return [a[:k] for a in sorted_dist_matrix]


def get_n_neighbors(rbg):
    """
    Description:
        Gets the array containing the amount of neighbors of each object.
    
    Parameters:
        rbg (2d Array of tuples (dist, index)): 2d Array corresponding to RBG.

    Returns:
        (Array[Int]): Array containing the amount of neighbors of each object.
    """
    return [len(a) for a in rbg]


def to_adj_m(neighbors):
    """
    Description:
        Transforms a neighbors list notation into an adjnacency matrix notation.
    
    Parameters:
        neighbors (2d Array of tuples (dist, index)): 2d Array representing a KNN or RBG in the neighbors list notation.
    
    Returns:
        (2d Array[Double]): Adjacency matrix containing the weight of the edge
    """
    adj_m = np.zeros(len(neighbors) ** 2).reshape((len(neighbors), len(neighbors)))
    for i, l in enumerate(neighbors):
        for a in l:
            adj_m[i][a[1]] = a[0]
            adj_m[a[1]][i] = a[0]
    return adj_m


def get_mst(graph):
    """
    Description:
        Computes the MST of a graph.
    
    Parameters:
        graph (2d Array[Double]): graph in adjacency matrix notation.
    
    Returns:
        (Scipy.sparse.csr_matrix): Sparse adjacency matrix containing the MST edges.
    """
    return minimum_spanning_tree(csr_matrix(graph))


def plot_sorted_n_neighbors(n_neighbors):
    """
    Description:
        Plots a bar chart representing (in a sorted way) the number of neighbors of each object.
    
    Parameters:
        n_neighbors (Array[Int]): Array containing the amount of neighbors of each object.
    """
    plt.bar(range(len(n_neighbors)),sorted(n_neighbors, reverse=True))
    
    
def plot_dataset(dataset):
    """
    Description:
        Plots a 2-dimensional dataset.
    
    Parameters:
        dataset (nd-Array of shape(N, 2)): Dataset where each row is an Array containing the x and y coordinates.
    """
    plt.scatter([a[0] for a in dataset], [a[1] for a in dataset])
    
    
def plot_radius(dataset, radius, c):
    """
    Description:
        Plots circles centered on the points in the data set and with predetermined radii and colors.
    
    Parameters:
        dataset (nd-Array of shape(N, 2)): Dataset where each row is an Array containing the x and y coordinates.
        radius (Array[Double]): List of radii
        c (Array[Int]): List of colors
    """
    x = [a[0] for a in dataset]
    y = [a[1] for a in dataset]
    assert len(x) == len(y) and len(y) == len(radius)
    circles(x, y, radius, c=c, alpha=0.2, ec='black')

    
def count_edges(graph):
    """
    Description:
        Computes the correct amount of edges.
    
    Parameters:
        graph (2d Array of tuples (dist, index)): 2d Array representing a graph in the neighbors list notation.
    
    Returns:
        (Int): Ammout of edges.
    """
    m = to_adj_m(graph)
    counter = sum([len(a[a != 0]) for a in m])
    return int(counter/2)


def get_edges(graph):
    """
    Description:
        Gets the list of edges contained in the graph in the adjacency matrix notation.
    
    Parameters:
        graph (2d Array[Double]): Adjacency matrix
    
    Returns:
        (Array of tuple (obj1, obj2)): array containing tuples (obj1, obj2) edges where index obj1 < obj2.
        
    Note: It can contain duplicated edges.
    """
    edges = [[(min(i, j), max(i, j)) for j, col in enumerate(row) if col != 0] for i, row in enumerate(graph)]
    return [col for row in edges for col in row]


def plot_edges(dataset, graph):
    """
    Description:
        Plots the edges in the plane.
    
    Parameters:
        dataset (nd-Array of shape(N, 2)): Dataset where each row is an Array containing the x and y coordinates.
        graph (2d Array[Double]): Adjacency matrix
    """
    edges = list(set(get_edges(graph)))
    for edge in edges:
        plt.plot([dataset[edge[0]][0], dataset[edge[1]][0]],[dataset[edge[0]][1], dataset[edge[1]][1]], c="black")


class DisjointSet: 
    def __init__(self, n): 
        self.rank = [1] * n 
        self.parent = [(i, 0) for i in range(n)] 
  
    def find(self, x): 
        if (self.parent[x][0] != x): 
            self.parent[x] = self.find(self.parent[x][0])
        return self.parent[x] 

    def union(self, x, y): 
        xset, xfrozen = self.find(x) 
        yset, yfrozen = self.find(y) 

        if xset == yset: 
            return

        if self.rank[xset] < self.rank[yset]: 
            self.parent[xset] = (yset, yfrozen | xfrozen)
  
        elif self.rank[xset] > self.rank[yset]: 
            self.parent[yset] = (xset, yfrozen | xfrozen) 
        else: 
            self.parent[yset] = (xset, yfrozen | xfrozen) 
            self.rank[xset] = self.rank[xset] + 1
            
        self.parent[xset] = (self.parent[xset][0], yfrozen | xfrozen)
        self.parent[yset] = (self.parent[yset][0], yfrozen | xfrozen)
            
    def freeze(self, x):
        p, frozen = self.find(x)
        if not frozen:
            self.parent[p] = (p, 1)

            
def f_boruvka(graph, k, duplicate_edges=False):
    cgraph = deepcopy(graph)
    if duplicate_edges:
        for i, row in enumerate(cgraph):
            for edge in row:
                cgraph[edge[1]].append((edge[0], i))
        cgraph = [list(set(row)) for row in cgraph]

    ds = DisjointSet(len(cgraph))
    mst = np.zeros(len(graph)**2).reshape((len(graph), len(graph)))
    k_count = [0] * len(graph)
    
    def remove_non_candidates_by_id(i):
        row = cgraph[i]
        iset, _ = ds.find(i)
        while len(row) and ds.find(row[0][1])[0] == iset:
            row.pop(0)
            k_count[i] += 1
            if k_count[i] == k:
                ds.freeze(i)
        if len(row) == 0:
            ds.freeze(i)

    def remove_non_candidates():
        for i in range(len(cgraph)):
            remove_non_candidates_by_id(i)
            
    def get_candidate_by_id(i):
        if len(cgraph[i]) == 0:
            return
        nc = cgraph[i][0]
        id_min = min(i, nc[1])
        id_max = i + nc[1] - id_min
        # (W, id_min, id_max, source)
        return (nc[0],id_min, id_max, i)
    
    def get_candidates():
        return [get_candidate_by_id(i) for i in range(len(cgraph)) if len(cgraph[i]) > 0 and not ds.find(i)[1]]
    
    discarted = []
    remove_non_candidates()
    candidates = get_candidates()
    heapify(candidates)
    while len(candidates):
        candidate = heappop(candidates)
        xset, xfrozen = ds.find(candidate[1])
        yset, yfrozen = ds.find(candidate[2])
        frozen = xfrozen if candidate[3] == candidate[1] else yfrozen
        iset = xset if candidate[3] == candidate[1] else yset
        if xset != yset:
            if not frozen:
                mst[candidate[1]][candidate[2]] = candidate[0]
                mst[candidate[2]][candidate[1]] = candidate[0]
                ds.union(candidate[1], candidate[2])
        remove_non_candidates_by_id(candidate[3])
        if len(cgraph[candidate[3]]):
            if not ds.find(candidate[3])[1]:
                heappush(candidates, get_candidate_by_id(candidate[3]))
    candidates = [[(item[0], min(item[1], i), max(item[1], i)) for item in row] for i, row in enumerate(cgraph)]
    candidates = [item for row in candidates for item in row]
    candidates = sorted(list(set(candidates)), reverse=True)
    while len(candidates):
        candidate = candidates.pop()
        xset, xfrozen = ds.find(candidate[1])
        yset, yfrozen = ds.find(candidate[2])
        if xset != yset:
            mst[candidate[1]][candidate[2]] = candidate[0]
            mst[candidate[2]][candidate[1]] = candidate[0]
            ds.union(candidate[1], candidate[2])
    return mst


def get_df_analysis(dataset_name, k_interval, types=["KNN", "KNN-F", "KNN-FD", "RBG"]):
    dataset = pd.read_csv(dataset_name, header=None).values
    sorted_dist_m = get_sorted_dist_matrix(dataset)
    mst = get_mst(to_adj_m(sorted_dist_m))
    mst_edges = set(get_edges(mst.toarray()))
    values = []
    # [type, K, n_edges, connected_components, wrong_mst_edges, total_mst_edges, missing_edges, acc, n_trees, mst]
    values.append([dataset_name, 'MST', 0, int((len(dataset) - 1)**2), 1, 0, len(mst_edges), 0, 1, 1, mst_edges])
    for k in k_interval:
        if "KNN" in types or "KNN-F" in types or "KNN-FD" in types:
            knn = get_knn(sorted_dist_m, k)
        if "RBG" in types:
            limit, rbg = get_rbg(sorted_dist_m, k)
        
        if "KNN" in types:
            knn_mst = get_mst(to_adj_m(knn))
            knn_mst_edges = set(get_edges(knn_mst.toarray()))
            values.append([dataset_name,
                       'KNN',
                       k,
                       int(count_edges(knn)),
                       connected_components(to_adj_m(knn))[0],
                       len(knn_mst_edges - mst_edges),
                       len(knn_mst_edges),
                       len(mst_edges - knn_mst_edges),
                       1 - len(knn_mst_edges - mst_edges)/len(knn_mst_edges),
                       connected_components(knn_mst)[0],
                       knn_mst_edges])
            
        if "KNN-F" in types:
            knn_fmst = f_boruvka(knn)
            knn_fmst_edges = set(get_edges(knn_fmst))
            values.append([dataset_name,
                       'KNN-F',
                       k,
                       int(count_edges(knn)),
                       connected_components(to_adj_m(knn))[0],
                       len(knn_fmst_edges - mst_edges),
                       len(knn_fmst_edges),
                       len(mst_edges - knn_fmst_edges),
                       1 - len(knn_fmst_edges - mst_edges)/len(knn_fmst_edges),
                       connected_components(knn_fmst)[0],
                       knn_fmst_edges])
        
        if "KNN-FD" in types:
            knn_fmst_d = f_boruvka(knn, True)
            knn_fmst_d_edges = set(get_edges(knn_fmst_d))
            values.append([dataset_name,
                       'KNN-FD',
                       k,
                       int(count_edges(knn)),
                       connected_components(to_adj_m(knn))[0],
                       len(knn_fmst_d_edges - mst_edges),
                       len(knn_fmst_d_edges),
                       len(mst_edges - knn_fmst_d_edges),
                       1 - len(knn_fmst_d_edges - mst_edges)/len(knn_fmst_d_edges),
                       connected_components(knn_fmst_d)[0],
                       knn_fmst_d_edges])
        
        if "RBG" in types:
            rbg_mst = get_mst(to_adj_m(rbg))
            rbg_mst_edges = set(get_edges(rbg_mst.toarray()))
            values.append([dataset_name,
                       'RBG',
                       k,
                       int(count_edges(rbg)),
                       connected_components(to_adj_m(rbg))[0],
                       len(rbg_mst_edges - mst_edges),
                       len(rbg_mst_edges),
                       len(mst_edges - rbg_mst_edges),
                       1 - len(rbg_mst_edges - mst_edges)/len(rbg_mst_edges),
                       connected_components(rbg_mst)[0],
                       rbg_mst_edges])
        
    df = pd.DataFrame(values)
    df.columns = ["dataset", "type", "k", "no_edges", "connected_components", "wrong_mst_edges", "total_mst_edges", "missing_edges", "acc", "no_trees", "mst"]
    return df


def NNDescent(dataset, k, d=minkowski, sr=0.5, maxi=10, et=0.001):
    #p = multiprocessing.Pool(processes=njobs)
    
    def initialize_B(node):
        # sample(V, K)
        sampled_index = random.sample(list(range(node)) + list(range(node+1, len(dataset))), k=k)
        bi = [(-d(dataset[node], dataset[index]), index, True) for index in sampled_index]
        heapify(bi)
        return bi
    
    def get_old(Bv):
        old = [b for b in Bv if b[2] == False]
        return old
    
    def get_new(Bv):
        new = [b for b in Bv if b[2] == True]
        if len(new):
            sampled_new = random.sample(new, k=min(max(int(sr*k), 1), len(new)))
        else:
            sampled_new = []
        return sampled_new
    
    def mark_as_false(Bv):
        return [(a[0], a[1], False) for a in Bv]
    
    def get_R(B):
        transformed = [[(a[1], a[0], x, a[2]) for a in B[x]] for x in range(len(dataset))]
        flatten = sorted([element for row in transformed for element in row])
        R = [[] for _ in range(len(dataset))] 
        for i, group in groupby(flatten, key=lambda x : x[0]):
            R[i] = [(a[1], a[2], a[3]) for a in group]
        return R
    
    def v_rv_union(V, RV):
        nV = list(set(V) | set(random.sample(RV, k=min(len(RV), max(int(sr*k), 1)))))
        return nV
    
    def update_Bv(B, v, new_item):
        if new_item[1] == v:
            return 0
        for item in B[v]:
            if item[1] == new_item[1]:
                return 0
        old_item = heappushpop(B[v], new_item)
        duplicated = False
        for item in remaining_edges[v]:
            if item[1] == old_item[1]:
                duplicated = True
        if not duplicated:
            heappush(remaining_edges[v], old_item)
        if old_item[2] == True:
            return 0
        return 1
    
    # B[v] = Sample(V, K)×{<inf, true>} ∀ v ∈ V
    B = [initialize_B(x) for x in range(len(dataset))]
    remaining_edges = [[] for _ in range(len(dataset))]
    et_threshold = et * len(dataset) * k
    c = et_threshold + 1
    iteration = 0
    while c >= et_threshold and iteration < maxi:
        old = [get_old(Bv) for Bv in B]
        new = [get_new(Bv) for Bv in B]
        B = [mark_as_false(Bv) for Bv in B]
        
        Rold = get_R(old)
        Rnew = get_R(new)
        
        old = [v_rv_union(old[x], Rold[x]) for x in range(len(dataset))]
        new = [v_rv_union(new[x], Rnew[x]) for x in range(len(dataset))]
        
        c = 0
        for v in range(len(dataset)): 
            for u1 in new[v]:
                for u2 in new[v]:
                    if u1[1] >= u2[1]:
                        continue
                    l = d(dataset[u1[1]], dataset[u2[1]])
                    c += update_Bv(B, u1[1], (-l, u2[1], True))
                    c += update_Bv(B, u2[1], (-l, u1[1], True))
                for u2 in old[v]:
                    l = d(dataset[u1[1]], dataset[u2[1]])
                    c += update_Bv(B, u1[1], (-l, u2[1], True))
                    c += update_Bv(B, u2[1], (-l, u1[1], True))
        iteration += 1
    B = [sorted([(-item[0], item[1]) for item in Bv]) for Bv in B]
    remaining_edges = [sorted([(-item[0], item[1]) for item in Bv]) for Bv in remaining_edges]
    return B, remaining_edges

class HDBSCAN_new:
    def __init__(self, dataset, mpts=2, mcls=1, dm_method='knn', mst_method='std', n_samples=5, use_remaining_edges=False,  pre_calc_nnd=None):
        assert dm_method == 'knn' or dm_method == 'maxr' or dm_method == 'nnd'
        assert mst_method == 'std' or mst_method == 'frozen'
        self.mpts = mpts-1
        self.mcls = mcls+1
        self.d_g = None
        self.core_d = None
        self.dmr_g = None
        self.mst = None
        self.mstx = None
        self.dataset = dataset
        self.hierarchy = None
        self.condensed_tree = None
        self.labels = None
        self.dm_method = dm_method
        self.mst_method = mst_method
        self.limit = None
        self.n_samples = n_samples
        self.maxd = None
        self.remaining = None
        self.use_remaining_edges = use_remaining_edges
        self.n_connect_edges = 0
        self.pre_calc_nnd = pre_calc_nnd
        
    def calc_d_g(self):
        if self.dm_method == 'knn':
            sorted_m = get_sorted_dist_matrix(self.dataset)
            d_g = get_knn(sorted_dist_matrix=sorted_m, k=self.mpts)
        elif self.dm_method == 'maxr':
            sorted_m = get_sorted_dist_matrix(self.dataset)
            self.limit, d_g = get_rbg(sorted_dist_matrix=sorted_m, min_k=self.mpts)
        elif self.dm_method == 'nnd':
            if self.pre_calc_nnd is None:
                d_g, self.remaining = NNDescent(self.dataset, self.mpts, sr=0.5)
            else:
                d_g, self.remaining = self.pre_calc_nnd
        self.d_g = d_g
        
    def calc_core_d(self):
        self.core_d = [row[self.mpts-1][0] for row in self.d_g]
        
    def calc_dmr_g(self):
        self.dmr_g = [sorted([(max(e[0], self.core_d[e[1]], self.core_d[i]), e[1]) for e in row]) for i, row in enumerate(self.d_g)]
    
    def calc_mst(self):
        if self.remaining is not None and self.use_remaining_edges:
            self.dmr_g = [list(merge(Bv, self.remaining[i])) for i, Bv in enumerate(self.dmr_g)]
        if self.mst_method == 'frozen':
            self.mst = f_boruvka(self.dmr_g, k=self.mpts)
        elif self.mst_method == 'std':
            self.mst = get_mst(to_adj_m(self.dmr_g)).toarray()
        
    def calc_mstx(self):
        self.mstx = deepcopy(self.mst)
        for i in range(len(self.mstx)):
            self.mstx[i][i] = self.core_d[i]
            
    def connect_by_sampling(self):
        count, labels = connected_components(self.mst)
        samples = []
        if count == 1:
            return
        for i in range(count):
            samples.append(list(random.sample(list(np.where(labels == i)[0]), k=min(self.n_samples, len(list(np.where(labels == i)[0]))))))
        edges = []
        for i in range(len(samples)):
            for a in samples[i]:
                for k in range(i+1, len(samples)):
                    for b in samples[k]:
                        edges.append((minkowski(self.dataset[a], self.dataset[b]), min(a, b), max(a, b)))
        edges = sorted(edges, reverse=True)
        self.n_connect_edges += len(edges)
        while len(edges) and count > 1:
            candidate = edges.pop()
            if labels[candidate[1]] != labels[candidate[2]]:
                self.maxd = candidate[0]
                self.mst[candidate[1]][candidate[2]] = candidate[0]
                self.mst[candidate[2]][candidate[1]] = candidate[0]
                count, labels = connected_components(self.mst)
        
    def calc_hierarchy(self):
        if self.dm_method == 'knn' or self.dm_method == 'nnd':
            if self.maxd is None:
                maxd = max(self.core_d) + 1
            else:
                maxd = self.maxd + 1
        elif self.dm_method == 'maxr':
            maxd = self.limit + 1
        edges = []
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                if max(self.mst[i][j], self.mst[j][i]) != 0:
                    edges.append(max(self.mst[i][j], self.mst[j][i]))
                else:
                    edges.append(maxd)
        self.hierarchy = linkage(np.array(edges), method='single')
        #self.hierarchy = self.hierarchy[:len(set(u.get_edges(self.mst)))]
    
    def run(self):
        self.calc_d_g()
        self.calc_core_d()
        self.calc_dmr_g()
        self.calc_mst()
        #self.calc_mstx()
        self.connect_by_sampling()
        self.calc_hierarchy()
        (self.labels, self.probabilities, self.stabilities, self.condensed_tree,
             _) = hdbscan.hdbscan_._tree_to_labels(self.dataset,
                                                    self.hierarchy,
                                                    min_cluster_size=self.mcls,
                                                    allow_single_cluster=True,
                                                    match_reference_implementation=True)

        
def HDBSCAN_Descent_naive(dataset, mpts=2, mcls=1, n_samples=5, pre_calc_nnd=None):
    return HDBSCAN_new(dataset, mpts, mcls, dm_method='nnd', n_samples=n_samples, use_remaining_edges=True, pre_calc_nnd=pre_calc_nnd)


def HDBSCAN_Descent_frozen(dataset, mpts=2, mcls=1, n_samples=5, pre_calc_nnd=None):
    return HDBSCAN_new(dataset, mpts, mcls, dm_method='nnd', n_samples=n_samples, mst_method='frozen', use_remaining_edges=True,  pre_calc_nnd=pre_calc_nnd)


def HDBSCAN_MaxRadius(dataset, mpts=2, mcls=1, n_samples=5):
    return HDBSCAN_new(dataset, mpts, mcls, dm_method='maxr', n_samples=n_samples)


def plot_dendrogram(Z, labels, cut=0):
    roots = leaders(Z, np.asarray(labels).astype('i'))
    print("Roots", map(str, roots[0]))
    print("Roots", ', '.join(map(str, roots[0])))
    fig, ax1 = plt.subplots()
    plt.title('HDBSCAN*')
    plt.xlabel('mpts')
    plt.ylabel('distance')

    if cut > 0:
        partitioning = fcluster(Z, cut, criterion='distance')
        plt.axhline(y=cut, c='k')
    else:
        partitioning = labels + 1
    norm = colors.Normalize(0, partitioning.max())
    dflt_col = "#cccccc"
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else dflt_col if partitioning[x] == 0 else plt.cm.gist_rainbow(norm(partitioning[x])) for x in i12)
        link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col

    dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=6.,  # font size for the x axis labels
            labels=labels,
            count_sort=True,
            link_color_func=lambda x: colors.to_hex(link_cols[x]),
            above_threshold_color='grey'
        )
    plt.show()


class HDBSCAN:
    def __init__(self, dataset, mpts=2, mcls=1):
        self.mpts = mpts-1
        self.mcls = mcls+1
        self.dataset = dataset
        self.d_g = None
        self.core_d = None
        self.dmr_g = None
        self.condensed_tree = None
        self.labels = None
        
    def calc_d_g(self):
        self.d_g = distance_matrix(self.dataset, self.dataset)
    
    def calc_core_d(self):
        self.core_d = [sorted(row)[self.mpts] for row in self.d_g]
    
    def calc_dmr_g(self):
        self.dmr_g = [[max(d, self.core_d[j], self.core_d[i]) for j, d in enumerate(row)] for i, row in enumerate(self.d_g)]
        
    def calc_hierarchy(self):
        edges = []
        for i in range(len(self.dataset)):
            for j in range(i+1, len(self.dataset)):
                edges.append(self.dmr_g[i][j])
        self.hierarchy = linkage(np.array(edges), method='single')
    
    def run(self):
        self.calc_d_g()
        self.calc_core_d()
        self.calc_dmr_g()
        self.calc_hierarchy()
        (self.labels, self.probabilities, self.stabilities, self.condensed_tree,
             _) = hdbscan.hdbscan_._tree_to_labels(self.dataset,
                                                    self.hierarchy,
                                                    min_cluster_size=self.mcls,
                                                    allow_single_cluster=True,
                                                    match_reference_implementation=True)
    

    
