import sys
import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from hdbscan.hdbscan import HDBSCAN

def dbmsts(datafile, kmin = 1, kmax = 16, delimiter=' ', method='core', efficient=True):
    
    h = HDBSCAN(datafile, min_pts=kmax, delimiter=delimiter)

    if method == 'core':
        h._hdbscan_knn(kmin=kmin, kmax=kmax)

    if method == 'core_inc':
        h._hdbscan_knn_incremental(kmin=kmin, kmax=kmax)

    if method == 'core_star':
        h._hdbscan_coresg_star(kmin=kmin, kmax=kmax)

    if method == 'rng':
        h._hdbscan_rng(kmin=kmin, kmax=kmax, quick=True, efficient=efficient)

    if method == 'single':
        h.hdbscan(min_pts=kmax)

    if method == 'single_k':
        h.hdbscan_k(min_pts=kmax)

    if method == 'single_k_star':
        h.hdbscan_k_star(min_pts=kmax)

    return None       
