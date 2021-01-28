import sys
import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from hdbscan.hdbscan import HDBSCAN

def rng_hdbscan(datafile, kmin = 1, kmax = 16):

    print("Starting Experiments") 
    
    h = HDBSCAN(datafile, min_pts=kmax, delimiter=' ')

    h.hdbscan(kmin=kmin, kmax=kmax, method='rng')

    return None       
