import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

def prim(data, core_distances, min_pts, self_edges):

    n = len(data)

    n_edges = 2*n - 1 if self_edges else n - 1

    # keeps track of which points are attached to the tree.
    attached = np.zeros(n)

    # keeps track of the number of edges added so far.
    num_edges = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    # arrays to keep track of the shortest connection to each point.
    nearest_points = np.zeros(n_edges)
    nearest_distances  = np.full(n_edges, np.inf)

    while (num_edges < n - 1):

        # keeps track of the closest point to the tree.
        nearest_distance = np.inf
        nearest_point = -1

        # marks current point as attached
        attached[current_point] = 1

        # loops over the dataset to find the next point to attach.
        for neighbor in range(n):    
            if attached[neighbor]: continue

            d = max(distance.euclidean(data[current_point], data[neighbor]),
                core_distances[current_point, -1],
                core_distances[neighbor, -1])

            # updates the closese point to neigbor.
            if d < nearest_distances[neighbor]:
                nearest_distances[neighbor] = d
                nearest_points[neighbor] = current_point
            
            # updates the closest point to the tree. 
            if nearest_distances[neighbor] < nearest_distance:
                nearest_distance = nearest_distances[neighbor]
                nearest_point = neighbor

        # attached nearest_point to the tree.

        current_point = nearest_point
        
        # updates the number of edges added.
        num_edges += 1

    # if self_edges:
        # nearest_points[n-1:] = np.arange(n)
        # nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
    
    return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))
