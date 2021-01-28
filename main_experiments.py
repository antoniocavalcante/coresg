import sys

from experiments import run

# from mst import mst

if __name__ == "__main__":
    
    run.rng_hdbscan(sys.argv[1], kmin=2, kmax=4)

    