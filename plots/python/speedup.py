from __future__ import division

import sys
import numpy as np

from scipy.stats.mstats import gmean

filename = "../results/ihdbscan-dataset.results.avg"

def name(exp, smart, naive):
    output="../results/ihdbscan" + "-" + exp

    if smart:
        output = output + "-smart"

    if naive:
        output = output + "-naive"

    return output + ".results.avg"

# Read core-distances times.
def readihdbscan(exp, smart, naive):

    ihdbscan = np.empty(7, dtype=long)
    i = 0

    fileihdbscan = name(exp, smart, naive)

    with open(fileihdbscan, 'r') as f:
        for line in f:
            splitLine = line.split()
            ihdbscan[i] = splitLine[6]
            i = i + 1

    hdbscan = np.empty(7, dtype=long)
    i = 0

    filehdbscan = "../results/hdbscan-" + exp + ".results.avg"

    with open(filehdbscan, 'r') as f:
        for line in f:
            splitLine = line.split()
            hdbscan[i] = splitLine[5]
            i = i + 1


    print hdbscan
    print ihdbscan

    print hdbscan/ihdbscan

    print gmean(hdbscan/ihdbscan, axis=0)

exp = "minpoints"
print "SPEEDUP MINPOINTS"
readihdbscan(exp, False, False)
readihdbscan(exp, True, False)
readihdbscan(exp, True, True)

exp = "dataset"
print "SPEEDUP DATASET"
readihdbscan(exp, False, False)
readihdbscan(exp, True, False)
readihdbscan(exp, True, True)

exp = "dimensions"
print "SPEEDUP DIMENSIONS"
readihdbscan(exp, False, False)
readihdbscan(exp, True, False)
readihdbscan(exp, True, True)
