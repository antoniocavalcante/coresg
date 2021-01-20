import numpy as np

from scipy.spatial import distance
from scipy.sparse import coo_matrix

def prim(data, core_distances, min_pts, self_edges):

    n = len(data)

    # keeps track of which points are attached to the tree.
    attached = np.zeros(n)

    # keeps track of the number of edges added so far.
    num_edges = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    # arrays to keep track of the shortest connection to each point.
    nearest_points = np.zeros(n-1)
    nearest_distances  = np.full(n-1, np.inf)

    while (num_edges < n - 1):

        # keeps track of the closest point to the tree.
        nearest_distance = -1
        nearest_point = -1

        # marks current point as attached
        attached[current_point] = 1

        # loops over the dataset to find the next point to attach.
        for neighbor in range(n):    
            if attached(neighbor): continue

            distance = distance.euclidean(data[current_point], data[neighbor])

            # updates the closese point to neigbor.
            if distance < nearest_distances[neighbor]:
                nearest_distances[neighbor] = distance
                nearest_points[neighbor] = current_point
            
            # updates the closest point to the tree. 
            if nearest_distances[neighbor] < nearest_distance:
                nearest_distance = nearest_distances[neighbor]
                nearest_point = neighbor

        # attached nearest_point to the tree.

        current_point = nearest_point
        
        # updates the number of edges added.
        num_edges += 1

    if self_edges:
        nearest_points[n:] = np.arange(n)

    return coo_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))
