import numpy as np
cimport numpy as np

cimport cython

from disjoint_set import DisjointSet
from scipy.spatial import distance
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from libc.math cimport sqrt

import heapq

include '../parameters.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim(
	DTYPE_t[:, :] data, 
	DTYPE_t[:] core_distances, 
	ITYPE_t self_edges):

	cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor
	cdef DTYPE_t nearest_distance, d

	n = len(data)

	n_edges = 2*n - 1 if self_edges else n - 1

	# keeps track of which points are attached to the tree.
	cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

	# arrays to keep track of the shortest connection to each point.
	cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
	cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

	cdef DTYPE_t[:] distances_array 

	a_knn = dok_matrix((n, n), dtype=DTYPE)

	# keeps track of the number of edges added so far.
	num_edges_attached = 0

	# sets current point to the last point in the data.
	current_point = n - 1

	while (num_edges_attached < n - 1):

		# keeps track of the closest point to the tree.
		nearest_distance = float("inf")
		nearest_point = -1

		# marks current point as attached
		attached[current_point] = 1

		distances_array = distance.cdist([data[current_point]], data)[0]

		# loops over the dataset to find the next point to attach.
		for neighbor in xrange(n):    
			if attached[neighbor]: continue

			d = max(
				distances_array[neighbor],
				core_distances[current_point], 
				core_distances[neighbor])

			if d == core_distances[current_point]:
				a_knn[current_point, neighbor] = d
				a_knn[neighbor, current_point] = d

			# updates the closese point to neigbor.
			if d < nearest_distances[neighbor]:
				nearest_distances[neighbor] = d
				nearest_points[neighbor] = current_point
			
			# updates the closest point to the tree. 
			if nearest_distances[neighbor] < nearest_distance:
				nearest_distance = nearest_distances[neighbor]
				nearest_point = neighbor

		# attached nearest_point to the tree.
		current_point = nearest_point
		
		# updates the number of edges added.
		num_edges_attached += 1

	# if self_edges:
		# nearest_points[n-1:] = np.arange(n)
		# nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
	
	return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n)), a_knn



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _prim_graph(
	DTYPE_t[:, :] data,
	graph,
	ITYPE_t[:, :] knn, 
	DTYPE_t[:, :] core_distances, 
	ITYPE_t min_pts, 
	ITYPE_t self_edges):

	cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor, i
	cdef DTYPE_t nearest_distance, d

	n = len(knn)

	n_edges = 2*n - 1 if self_edges else n - 1

	# keeps track of which points are attached to the tree.
	cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

	# arrays to keep track of the shortest connection to each point.
	cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
	cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

	cdef DTYPE_t[:] distances_array 

	# keeps track of the number of edges added so far.
	num_edges_attached = 0

	# sets current point to the last point in the data.
	current_point = n - 1

	pq = []

	# heapq.heapify(pq)
	
	heapq.heappush(pq, (0, current_point))

	while (num_edges_attached < n - 1):

		# retrieves the closest point to the tree.
		_, current_point = heapq.heappop(pq)

		# attaches current_point and marks it as attached.
		attached[current_point] = 1

		# loops over the k-NNG.
		for i in range(1, min_pts + 1):

			neighbor = knn[current_point, i]
			
			if attached[neighbor]: continue

			d = max(
				core_distances[current_point, i],
				core_distances[current_point, min_pts], 
				core_distances[neighbor, min_pts])

			# updates the closese point to neighbor.
			if d < nearest_distances[neighbor]:
				nearest_distances[neighbor] = d
				nearest_points[neighbor] = current_point
				heapq.heappush(pq, (d, neighbor))		

		# loops over the MST.
		for i in range(len(graph.rows[current_point])):
			neighbor = graph.rows[current_point][i]

			if attached[neighbor]: continue

			d = max(
				euclidean_local(data[current_point], data[neighbor]),
				core_distances[current_point, min_pts], 
				core_distances[neighbor, min_pts])

			if d < nearest_distances[neighbor]:
				nearest_distances[neighbor] = d
				nearest_points[neighbor] = current_point
				heapq.heappush(pq, (d, neighbor))
		
		# updates the number of edges added.
		num_edges_attached += 1

	# if self_edges:
		# nearest_points[n-1:] = np.arange(n)
		# nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
	
	return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim_order(
	DTYPE_t[:] data,
	int[:] indices,
	int[:] indptr,
	int n):

	cdef ITYPE_t n_edges, num_points_attached, current_point, nearest_point, neighbor, i
	cdef DTYPE_t nearest_distance, d, weight

	n_edges = n - 1

	# keeps track of which points are attached to the tree.
	cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

	# arrays to keep track of the shortest connection to each point.
	cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
	cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

	cdef DTYPE_t[:] distances_array 

	cdef ITYPE_t[:] order_p = np.zeros(n_edges, dtype=ITYPE)
	cdef DTYPE_t[:] order_w = np.zeros(n_edges, dtype=DTYPE)

	# keeps track of the number of edges added so far.
	num_points_attached = 0

	# sets current point to the last point in the data.
	current_point = n - 1

	pq = []
	
	heapq.heappush(pq, (0, current_point))

	while (num_points_attached < n - 1):

		# retrieves the closest point to the tree.
		weight, current_point = heapq.heappop(pq)

		order_p[num_points_attached] = current_point
		order_w[num_points_attached] = weight

		# attaches current_point and marks it as attached.
		attached[current_point] = 1

		# loops over the MST.
		for i in xrange(indptr[current_point], indptr[current_point+1]):
			neighbor = indices[i]

			if attached[neighbor]: continue

			d = data[i]

			if d < nearest_distances[neighbor]:
				nearest_distances[neighbor] = d
				nearest_points[neighbor] = current_point
				heapq.heappush(pq, (d, neighbor))
		
		# updates the number of edges added.
		num_points_attached += 1
	
	return order_p, order_w



#MST calculation with initial edges set by 1-knng
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _split_mst(data, nn_distances, knn, mpts, self_edges):

	#auxiliary data structures
	#data size
	n = len(data)
	#disjoint set
	ds = DisjointSet()
	#creating matrix of mrd distances
	mrd = np.zeros((n,mpts))
	#matrix for MST edges
	mst_edges = np.zeros((n-1,2), dtype=int)
	#array of MST weights
	mst_weights = np.zeros(n-1)
	#edges count
	nedges = 0


	#init MST with closest neighbour
	for i in range(n): 
		#flag that indicates connection
		connected = False
		#calculating mrd for all points and building 1-NNG
		mrd[i,0] = nn_distances[i,mpts-1]
		for neighbor in range(1,mpts):
			mrd[i,neighbor] = max(nn_distances[i,neighbor],
			nn_distances[i, mpts-1],
			nn_distances[knn[i,neighbor], mpts-1])
			#building 1-NNG
			if (mrd[i,neighbor] == mrd[i,0] and not(connected)):
				if not(ds.connected(i,knn[i,neighbor])):
					ds.union(i,knn[i,neighbor])
					mst_edges[nedges] = (i,knn[i,neighbor])
					mst_weights[nedges] = mrd[i,neighbor]
					nedges += 1
					connected = True
		#insert singleton point in sets if not connected			
		if not(connected): 
			ds.find(i)		
						
	#Updating sets info
	sets = list(ds.itersets()) 
	nsets = len(sets)
	inter_mrd = np.matrix(np.ones((nsets,nsets)) * np.inf)
	#Updating self-edges
	for i in range(nsets):
		inter_mrd[i,i] = 0	
	inter_edges = np.zeros((nsets,nsets,2), dtype=int)

	#Inter-sets distances calculation and selection
	for i in range(nsets):
		for j in range(i+1,nsets):
			#conveting sets to lists
			setlist1 = list(sets[i])
			setlist2 = list(sets[j])
			D = distance.cdist(data[setlist1],data[setlist2],'euclidean')
			#finding lowest mrd inter-set 
			for l in range(len(setlist1)):
				for m in range(len(setlist2)):
					aux = max(D[l,m], mrd[setlist1[l],0], mrd[setlist2[m],0])
					if inter_mrd[i,j] > aux:
						inter_mrd[i,j] = aux
						inter_mrd[j,i] = aux
						inter_edges[i,j] = (setlist1[l],setlist2[m])
						inter_edges[j,i] = inter_edges[i,j]
															
	#building MST of the initial sets
	inter_mst = minimum_spanning_tree(inter_mrd)

	#converting interset MST into global MST
	(ind1,ind2) = inter_mst.nonzero()
	for i in range(len(ind1)):
		mst_edges[nedges] = inter_edges[ind1[i],ind2[i]]
		mst_weights[nedges] = inter_mrd[ind1[i],ind2[i]]
		nedges += 1
		
	#temporary debug
	if nedges < (n-1):
		print("Faltou aresta " + str(nedges))
		
	#building and returning MST matrix
	return csr_matrix((mst_weights, (mst_edges[:,0], mst_edges[:,1])), shape=(n, n))


# cpdef prim(DTYPE_t[:, :] data, DTYPE_t[:] core_distances, np.int64_t self_edges):
# 	return _prim(data, core_distances, self_edges)


cpdef prim_graph(DTYPE_t[:, :] data, graph, knn, DTYPE_t[:, :] core_distances, np.int64_t min_pts, np.int64_t self_edges):
	return _prim_graph(data, graph, knn, core_distances, min_pts, self_edges)


# cpdef prim_order(mst):
# 	return _prim_order(mst)

cpdef split_mst(data, nn_distances, knn, mpts, self_edges):
	return _split_mst(data, nn_distances, knn, mpts, self_edges)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef DTYPE_t euclidean_local(DTYPE_t[:] v1, DTYPE_t[:] v2):
	cdef ITYPE_t i, m
	cdef DTYPE_t d = 0.0
	m = v1.shape[0]

	for i in xrange(m):
		d += (v1[i] - v2[i])**2

	return sqrt(d)
