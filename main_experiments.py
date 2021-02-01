import pyximport; 
pyximport.install()

import time

import sys

from experiments import run

if __name__ == "__main__":
    
    print(sys.argv[1], sys.argv[2], sep=' ', end=' ')

    start = time.time()
    run.g_hdbscan(datafile=sys.argv[1], kmax=int(sys.argv[2]), delimiter=sys.argv[3], method=sys.argv[4])
    end = time.time()
    print(str(end - start))