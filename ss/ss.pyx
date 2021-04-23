import pyximport
pyximport.install()

import numpy as np
cimport numpy as np

cimport cython

include '../parameters.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef label_expansion(
    int[:] mst_indices,
    int[:] mst_indptr,
    DTYPE_t[:] mst_data,
    int[:] labels):


    cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor, i, n_current
    cdef DTYPE_t nearest_distance, d

    n = data.shape[0]

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    cdef DTYPE_t[:] distances_array 

    cdef FibonacciHeap heap
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(n * sizeof(FibonacciNode))

    for i in xrange(n):
        initialize_node(&nodes[i], i)

    cdef FibonacciNode *v
    cdef FibonacciNode *current_neighbor

    for p in xrange(n):

        if labels[p] != -1:

            for i in xrange(mst_indptr[current_point], mst_indptr[current_point+1]):
                current_neighbor = &nodes[mst_indices[i]]



    return labels