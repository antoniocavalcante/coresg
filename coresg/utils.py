from sklearn.neighbors import kneighbors_graph
import pynndescent

from random import seed
from random import sample
from time import monotonic

from scipy.spatial import distance_matrix

from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix

import matplotlib.pyplot as plt

def convert_pynndescent_output(neighbors):

    n = len(neighbors[0])
    k = len(neighbors[0][0])-1
    
    knn = [[(neighbors[1][i][j], neighbors[0][i][j]) for j in range(k+1)] for i in range(n)]
    
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
    plt.scatter([a[0] for a in dataset], [a[1] for a in dataset], s=10)
    
    
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


def hai(h1, h2):
    
    assert len(h1[0]) == len(h2[0])
    
    num_of_objects = len(h1[0])
    fact = 1/(num_of_objects**2)
    inter = 0
    
    for i in range(0, num_of_objects):
        for j in range(0, num_of_objects):
            if (i==j):
                continue
            s1 = find(i, j, h1)/ num_of_objects
            s2 = find(i, j, h2)/ num_of_objects
            inter = inter + abs(s1-s2)
            
    return 1-(fact*inter)



def build_knng(data, k, method='sklearn'):
    if method == 'sklearn':
        return kneighbors_graph(data, k, mode='distance', include_self=False)
    
    elif method == 'pynndescent':        
        index = pynndescent.NNDescent(data, n_neighbors = k, tree_init = False, verbose=True)
        return convert_pynndescent_output(index.neighbor_graph)

    elif method == 'scann':
        return kneighbors_graph(data, k, mode='distance', include_self=False)
    
    else:
        raise ValueError('This method for finding the k-NNG is not supported.')
        
        

#definition of method that builds the Core-SG
def buildCoreSG(knn, mst):
    
    #Building Core-SG
    CoreSG = [[nn for nn in nnlist] for nnlist in knn]
    
    #Adding edges 
    rows,cols = mst.nonzero()
    for row,col in zip(rows,cols):
        CoreSG[row] = CoreSG[row] + [(mst[row,col], col)]
        CoreSG[col] = CoreSG[col] + [(mst[row,col], row)]
        
    #Eliminating duplicates and sorting
    for i in range(len(CoreSG)):
        edges_set = set(CoreSG[i])
        CoreSG[i] = list(edges_set)
        CoreSG[i].sort()
    
    return CoreSG
        

# def merge_edges(graph1, graph2, index):
    
#     #converting graph 2 labels into graph 1 labels 
#     converted_graph2 = [[nn for nn in nnlist] for nnlist in graph2] 
#     for i in range(len(graph2)):
#         for j in range(len(graph2[i])): 
#             converted_graph2[i][j] = (graph2[i][j][0],index[graph2[i][j][1]])
    
#     #concatenate graph2 in graph1, delete repetitions, sort edges
#     final_graph = [[nn for nn in nnlist] for nnlist in graph1] 
#     for i in range(len(index)):
#         final_graph[index[i]] = final_graph[index[i]] + converted_graph2[i]
#         edges_set = set(final_graph[index[i]])
#         final_graph[index[i]] = list(edges_set)
#         final_graph[index[i]].sort()
    
#     return(final_graph)
    
    
def merge_edges(g1, g2, index):
        
    aux = lil_matrix(g1.shape)
    
    rows, cols = g2.nonzero()
    
    for row, col in zip(rows, cols):
        aux[index[row], index[col]] = g2[row, col]
        
    return aux.maximum(g1)


def sample_data(data, sample_rate = 0.05):
    
    n = len(data)

    sample_size = int(n * sample_rate)
    
    seed(monotonic())
    sequence = range(n)
    idx = sample(sequence, sample_size)

    return data[idx,:], idx



def aux_graphs_sample(data, k):
    #Calculating MST and Core-SG of the sample
    sorted_dist_sample = get_sorted_dist_matrix(data)

    #Calculating Sample MST
    mst_sample = get_mst(to_adj_m(sorted_dist_sample))

    #Calculating KNN of samples
    knn_sample = get_knn(sorted_dist_sample, k) 

    #Building Core-SG of the sample
    return mst_sample, buildCoreSG(knn_sample, mst_sample)


#function that calculates mutual reachability distances
def calculate_mrd(sorted_distance_matrix, mpts):

    #calculating core distances
    core_dist = [sorted_distance_matrix[i][mpts-2][0] for i in range(len(sorted_distance_matrix))]
    
    #calculating mrd wrt mpts 
    mrd_dist = [[(max(sorted_distance_matrix[i][j][0],core_dist[i], core_dist[j]),sorted_distance_matrix[i][j][1]) for j in range(len(sorted_distance_matrix[i]))] for i in range(len(sorted_distance_matrix))]
    
    #sorting distances
    for i in range(len(mrd_dist)):
        mrd_dist[i].sort()

    return (mrd_dist, core_dist)


def find(i, j, h):
    for l in range(len(h)-1, -1, -1):
        if h[l][i] != 0 and (h[l][i] == h[l][j]):
            if(l != len(h)):
                c_no = int(h[l][i])
                count = 0
                
                for k in range(1, len(h[0])):
                    if int(h[l][k] == c_no):
                        count += 1
                return count
            else:
                return 0
    return 0

