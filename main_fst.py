import numpy as np

from sklearn.neighbors import NearestNeighbors

from mst import mst

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from rng.fair_split_tree import FairSplitTree


if __name__ == "__main__":
    
    # generates a small random dataset
    data = np.array([[2, 3, 1, 0],[4, 6, 1, 5], [7, 2, 6, 15], [7, 9, 9, 23], [3, 17, 14, 6], [4, 14, 0, 3], [0, 1, 9, 0]])
    
    print(data)

    import sys
    print(sys.getrecursionlimit())


    fst = FairSplitTree(data)