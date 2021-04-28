# import pyximport; 
# pyximport.install()

import sys
import time
# import argparse

from experiments import run

# parser = argparse.ArgumentParser()

# parser.add_argument("data", type=str, help="path to input file")
# parser.add_argument("kmax", type=int, help="maximum value of mpts to be used")
# parser.add_argument("sep", type=str, help="delimiter for parsing the CSV files")
# parser.add_argument("method", type=str, help="method knn, knn_inc, rng")
# parser.add_argument("efficient", type=bool, help="efficient computation of rng")


# args = parser.parse_args()

# print(args)

if __name__ == "__main__":

    print(sys.argv[1], sys.argv[2], sep=' ', end=' ', flush=True)

    start = time.time()

    run.g_hdbscan(
        datafile=sys.argv[1], 
        kmax=int(sys.argv[2]), 
        delimiter=sys.argv[3], 
        method=sys.argv[4],
        efficient=True)
    
    end = time.time()
    print(str(end - start))