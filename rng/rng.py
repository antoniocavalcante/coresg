import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

class RelativeNeighborhoodGraph:

    def __init__(self, data, quick = True, naive = False):
        self.data = data            

        n = len(data)

        self.u = []
        self.v = []
        self.w = []

        # Build Fair Split Tree

        # Find Well-Separated Pairs and their respective SBCN

        return csr_matrix((self.w, (self.u, self.v)), shape=(n, n))

    def sbcn(self, red, blue):

        # if both sets are singletons
        if len(red) == 1 and len(blue) == 1:
            self.add_edge(red[-1], blue[-1])
            return

        if len(red) == 1:
            min_distance = np.inf
            v = -1
            for i in blue:
                current_distance = distance.euclidean(data[red[-1]], data[i])
                
                if current_distance < min_distance:
                    v = i
                    min_distance = current_distance

            self.add_edge(red[-1], v)
            return 

        if len(blue) == 1:
            min_distance = np.inf
            v = -1
            for i in blue:
                current_distance = distance.euclidean(data[red[-1]], data[i])
                
                if current_distance < min_distance:
                    v = i
                    min_distance = current_distance

            self.add_edge(red[-1], v)
            return 
        
        temp = []
        for r in red:
            
            nearest_p = -1
            nearest_d = np.inf
            
            for b in blue:
                min_dist_rb = distance.euclidean(data[r], data[b])

            temp.append(nearest_p)


        self.add_edge()
        

    def wspd(self, fst):
        stack = []

        stack.append(fst.root)

        while stack:
            node = stack.pop()

            if node.l is not None:
                stack.append(node.l)

            if node.r is not None:
                stack.append(node.r) 

            self.find_pairs(node.l, node.r)

        return None


    def find_pairs(self, fst):

        return None


    def add_edge(self, p, q):

        if self.quick:
            pass

        if self.naive:
            pass


if __name__ == "__main__":
    
    # generates a small random dataset
    data = np.array([[2, 3, 1, 0],[4, 6, 1, 5], [7, 2, 6, 15], [7, 9, 9, 23], [3, 17, 14, 6], [4, 14, 0, 3], [0, 1, 9, 0]])
    
    print(data)

    import sys
    print(sys.getrecursionlimit())


    fst = fair_split_tree.FairSplitTree(data)
