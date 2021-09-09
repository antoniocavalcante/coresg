import numpy as np
import pandas as pd
import hdbscan


def get_single_linkage(dataset, mpts):
    hd = hdbscan.HDBSCAN(min_samples=mpts,
                match_reference_implementation=True,
                gen_min_span_tree=True,
                approx_min_span_tree=False,
                allow_single_cluster=True)
    dt = pd.read_csv(dataset, header=None).values
    hd.fit(dt)
    return np.rot90(hd.single_linkage_tree_.to_numpy())


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

def compare(h1, h2):
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

