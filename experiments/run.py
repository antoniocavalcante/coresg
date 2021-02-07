import sys
import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from hdbscan.hdbscan import HDBSCAN

def g_hdbscan(datafile, kmin = 1, kmax = 16, delimiter=' ', method='knn'):
    
    h = HDBSCAN(datafile, min_pts=kmax, delimiter=delimiter)

    h.hdbscan_g(kmin=kmin, kmax=kmax, method=method, quick=True)

    return None       
