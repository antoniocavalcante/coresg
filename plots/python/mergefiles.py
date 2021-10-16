import sys
import numpy as np

import pandas as pd

from natsort import index_natsorted

core = {}
hdbscan = {}

# Read core-distances times.
def readcore(filename):
    with open(filename, 'r') as f:
        for line in f:
            splitLine = line.split()
            core[splitLine[0]] = int(splitLine[-1])


def readinitial(filename, columns):

    data = pd.read_csv(filename, sep=" ", names=columns)

    data = data.groupby(['dataset'], as_index = False).mean()

    data.replace(to_replace ='^.*data.*\/',
                 value = '',
                 regex = True,
                 inplace = True)

    data.replace(to_replace ='d-.*.dat',
                 value = '',
                 regex = True,
                 inplace = True)

    data.sort_values(by = "dataset",
                     key = lambda x: np.argsort(index_natsorted(data["dataset"])),
                     inplace = True)

    data.to_csv(path_or_buf = filename + ".avg",
                index=False,
                sep = " ",
                header = True)


# Read IHDBSCAN times.
def readihdbscan(filename, columns):

    data = pd.read_csv(filename, sep=" ", names=columns)

    if '-dataset' in filename:
        data.replace(to_replace ='^.*data.*\/.*d-',
                     value = '',
                     regex = True,
                     inplace = True)
        data.replace(to_replace ='n-.*dat',
                     value = 'k',
                     regex = True,
                     inplace = True)

    if '-dimensions' in filename:
        data.replace(to_replace ='^.*data.*\/',
                     value = '',
                     regex = True,
                     inplace = True)
        data.replace(to_replace ='-.*.dat',
                     value = '',
                     regex = True,
                     inplace = True)

    if '-clusters' in filename:
        data.replace(to_replace ='^.*data.*\/.*-',
                     value = '',
                     regex = True,
                     inplace = True)
        data.replace(to_replace ='c\.dat',
                     value = '',
                     regex = True,
                     inplace = True)


    if '-minpoints' not in filename and '-speedup' not in filename:
        data = data.groupby(['dataset'], as_index = False).mean()

        data.sort_values(by = "dataset",
                         key = lambda x: np.argsort(index_natsorted(data["dataset"])),
                         inplace = True)
    else:
        data = data.groupby(['minpoints'], as_index = False).mean()
        data.sort_values(by = "minpoints",
                         key = lambda x: np.argsort(index_natsorted(data["minpoints"])),
                         inplace = True)


    data.to_csv(path_or_buf = filename + ".avg",
                index=False,
                sep = " ",
                header = True)


def readhdbscan(filename):
    with open(filename, 'r') as f:
        j = 0
        r = np.zeros((7,6), dtype=np.int)

        d = np.empty([7,1], dtype='S20')

        for line in f:

            data = line.strip().split(" ")[0]

            x = [int(l) for l in line.strip().split(" ")[1:]]

            r[j // 5, 1:] = np.add(r[j // 5, 1:], x)
            r[j // 5, 0 ] = core[data]*5
            d[j // 5] = data

            j = j + 1

    r = r/5

    r[:, -1] = r[:, -1] + r[:, 0]

    r = r.astype('str')
    p = np.concatenate((d, r), axis=1)

    np.savetxt(filename + ".avg", p, fmt='%s', delimiter=' ')   # X is an array


if len(sys.argv) < 2:
    print("Error: missing filename.")

columns = []

if "initial" in sys.argv[1]:
    print("Loading result files...")

    columns = ["dataset", "minpoints", "graph", "msts", "size", "total"]

    readinitial(sys.argv[1], columns)

    sys.exit()


if "mrg" in sys.argv[1]:
    columns = ["dataset", "minpoints", "run", "mst", "hierarchies", "total"]
elif "-rng-" in sys.argv[1]:
    columns = ["dataset", "minpoints", "graph", "msts", "size", "total"]
elif "-core-inc-" in sys.argv[1]:
    columns = ["dataset", "minpoints", "graph", "msts", "total"]
elif "-core-" in sys.argv[1]:
    columns = ["dataset", "minpoints", "graph", "msts", "size", "total"]

readihdbscan(sys.argv[1], columns)
