import sys
import numpy as np

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from .hdbscan.hdbscan import HDBSCAN

def experiments(datafile, kmin = 1, kmax = 16):

    print("Starting Experiments") 
    
    h = HDBSCAN(datafile)

    

    return None       



if __name__ == "__main__":
    experiments(sys.argv)