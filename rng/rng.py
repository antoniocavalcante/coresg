import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from fair_split_tree import FairSplitTree

class RelativeNeighborhoodGraph:

    def __init__(self, data, quick = True, naive = False):
        self.data = data

        self.quick = quick
        self.naive = naive

        n = len(data)

        self.u = []
        self.v = []
        self.w = []

        # Build Fair Split Tree
        T = FairSplitTree(self.data)

        # Find Well-Separated Pairs and their respective SBCN
        self.wspd(T)

        self.graph = csr_matrix((self.w, (self.u, self.v)), shape=(n, n))


    def sbcn(self, red, blue):
        print(red, blue)
        # if both sets are singletons
        if len(red) == 1 and len(blue) == 1:
            self.add_edge(red[-1], blue[-1])
            return
        
        candidate_edges  = []
        candidate_points = set()
        for r in red:
            
            nearest_p = []
            nearest_d = np.inf
            
            for b in blue:
                min_dist_rb = distance.euclidean(data[r], data[b])

                if min_dist_rb <  nearest_d:
                    nearest_p = []

                if min_dist_rb <= nearest_d:
                    nearest_d = min_dist_rb
                    nearest_p.append(b)

            for b in nearest_p:
                candidate_edges.append((r, b))
                candidate_points.add(b)

        for b in candidate_points:
            
            nearest_p = []
            nearest_d = np.inf

            for r in red:
                min_dist_br = distance.euclidean(data[r], data[b])

                if min_dist_br <  nearest_d:
                    nearest_p = []

                if min_dist_br <= nearest_d:
                    nearest_d = min_dist_br
                    nearest_p.append(r)

            for r in nearest_p:
                if (r, b) in candidate_edges:
                    self.add_edge(r, b)
                    candidate_edges.remove((r, b))

    def add_edge(self, point_a, point_b):
        if self.relative_neighbors(point_a, point_b):
            self.u.append(point_a)
            self.v.append(point_b)
            self.w.append(distance.euclidean(self.data[point_a], self.data[point_b]))


    def wspd(self, fst):
        stack = []

        stack.append(fst.root)

        while stack:
            node = stack.pop()

            if not node.l.leaf:
                stack.append(node.l)

            if not node.r.leaf:
                stack.append(node.r) 

            self.find_pairs(node.l, node.r)


    def find_pairs(self, node_a, node_b):

        if FairSplitTree.separated(node_a, node_b):
            self.sbcn(node_a.points, node_b.points)
        else:
            if node_a.diameter <= node_b.diameter:
                self.find_pairs(node_a, node_b.l)
                self.find_pairs(node_a, node_b.r)
            else:
                self.find_pairs(node_a.l, node_b)
                self.find_pairs(node_a.r, node_b)
    

    def relative_neighbors(self, point_a, point_b):

        if self.quick:
            if not self._relative_neighbors_quick(point_a, point_b):
                return False
            
        if self.naive:
            if not self._relative_neighbors_naive(point_a, point_b):
                return False

        return True


    def _relative_neighbors_quick(self, point_a, point_b):
        return True


    def _relative_neighbors_naive(self, point_a, point_b):

        distance_ab = distance.euclidean(self.data[point_a], self.data[point_b])

        for point_c in range(len(self.data)):
            if distance_ab > max(
                distance.euclidean(self.data[point_a], self.data[point_c]), 
                distance.euclidean(self.data[point_b], self.data[point_c])):
                return False

        return True




if __name__ == "__main__":
    
    # generates a small random dataset
    data = np.array([
        [0, 2],
        [1, 1], 
        [1, 2], 
        [1, 3], 
        [2, 2], 
        [3, 1], 
        [4, 5],
        [5, 4],
        [5, 5],
        [5, 6],
        [6, 7],
        [7, 1],
        [7, 2],
        [8, 1],
        [8, 2]])
    
    rng = RelativeNeighborhoodGraph(data, quick=False, naive=True)

    n = len(data)

    print(rng.graph)

    print("------------------------------------")

    u = []
    v = []
    w = []

    for i in range(n):
        for j in range(i + 1, n):
            dij = distance.euclidean(data[i], data[j])

            rn = True

            for m in range(n):
                dim = distance.euclidean(data[i], data[m])
                djm = distance.euclidean(data[j], data[m])
                if (dij > max(dim, djm)):
                    rn = False
                    break
            
            if rn:
                u.append(i)
                v.append(j)
                w.append(dij)

    graph = csr_matrix((w, (u, v)), shape=(n, n))
    print(graph)
