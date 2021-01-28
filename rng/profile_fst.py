import pstats, cProfile

import numpy as np

import pyximport
pyximport.install()

from fair_split_tree import FairSplitTree

data = np.random.rand(10000,2)

fst = FairSplitTree(data)

cProfile.runctx("fst.construct()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
