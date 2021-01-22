import numpy as np

from scipy.spatial import distance

class FairSplitTree:

    def __init__(self, data):
        self.data = data

        self.root = self.FairSplitTreeNode(np.arange(len(data)))

        self.construct()

    def construct(self):

        # starts with the root of the tree
        stack = [self.root]

        while stack:

            node = stack.pop()

            if len(node.points) > 1:
                maxdim = np.max(data[node.points], axis=0)
                mindim = np.min(data[node.points], axis=0)

                # updates the diameter of the node (euclidean distance)
                node.diameter = distance.euclidean(maxdim, mindim)
                
                # updates the geometric center of this node
                node.center = (maxdim + mindim)/2

                split_dim = np.argmax(maxdim - mindim)
                split_val = (mindim[split_dim] + maxdim[split_dim]) / 2
                
                left  = [point for point in node.points if data[point, split_dim] <  split_val]
                right = [point for point in node.points if data[point, split_dim] >= split_val]

                if (left):
                    node.l = FairSplitTree.FairSplitTreeNode(left)
                    stack.append(node.l)

                if (right):
                    node.r = FairSplitTree.FairSplitTreeNode(right)
                    stack.append(node.r)

                print(split_dim, split_val, left, right)
            else:
                node.center = node.points[0]    


    class FairSplitTreeNode:

        def __init__(self, points, left = None, right = None, center = None):
            self.points = points
            self.l = left
            self.r = right
            self.diameter = 0
            self.center


def separated(node_a, node_b):
    return node_distances(node_a, node_b) > max(node_a.diameter, node_b.diameter)


def node_distances(node_a, node_b):
    return distance.euclidean((node_a.center, node_b.center) - node_a.diameter/2 - node_b.diameter/2



if __name__ == "__main__":
    
    # generates a small random dataset
    data = np.array([[2, 3, 1, 0],[4, 6, 1, 5], [7, 2, 6, 15], [7, 9, 9, 23], [3, 17, 14, 6], [4, 14, 0, 3], [0, 1, 9, 0]])
    
    print(data)

    import sys
    print(sys.getrecursionlimit())


    fst = FairSplitTree(data)
