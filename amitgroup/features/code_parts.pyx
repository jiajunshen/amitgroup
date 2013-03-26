#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float64
UINT = np.uint8
UINT32 = np.uint32
ctypedef np.float64_t DTYPE_t
ctypedef np.uint8_t UINT_t
ctypedef np.uint32_t UINT32_t

cdef unsigned int _count_edges(UINT_t[:,:,:] X,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z) nogil:
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            for z in range(num_z):
                if X[i,j,z]:
                    count += 1
    return count



# cdef compute_loglikelihoods(np.ndarray[ndim=2,dtype=UINT_t] X,
#                            unsigned int i_start,
#                            unsigned int i_end,
#                            unsigned int j_start,
#                            unsigned int j_end,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_parts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_invparts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] out_map,
#                            unsigned int num_parts):
#     for i in range(i_end-i_start):
#         for j in range(j_end-j_start):
#             if X[i_start+i,j_start+j]:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_parts[k,i,j]
#             else:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_invparts[k,i,j]

cdef unsigned int _count_edges_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
                      np.ndarray[ndim=2,dtype=UINT_t] M,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z):
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if M[i,j]:
                for z in range(num_z):
                    if X[i,j,z]:
                        count += 1
    return count



def code_parts(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
               int threshold, outer_frame=0):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    #cdef UINT_t[:,:]
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = 1+i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = 1+j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]
            
                if count >= threshold:
                    out_map_mv[i_start,j_start] = 0.0
                    out_map_mv[i_start,j_start,0] = NINF 
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    for k in range(num_parts):
                                        out_map_mv[i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                                else:
                                    for k in range(num_parts):
                                        out_map_mv[i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                
    return out_map


def code_parts_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold, 
                    np.ndarray[ndim=2,dtype=UINT_t] M):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    M : ndarray[ndim=2]
        Mask over the patches that determines which X-Y locations will be used for computing whether the number of edges reaches a certain threshold. Should be a binary array
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef DTYPE_t[:,:,:] out_map_mv = out_map
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim
            count = _count_edges_mask(X, M,
                                            i_start,i_end,j_start,j_end,X_z_dim)

            if count >= threshold:
                out_map_mv[i_start,j_start] = 0.0
                out_map_mv[i_start,j_start,0] = NINF 
                for i in range(part_x_dim):
                    for j in range(part_y_dim):
                        for z in range(X_z_dim):
                            if X_mv[i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    out_map_mv[i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                            else:
                                for k in range(num_parts):
                                    out_map_mv[i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
            else:
                out_map_mv[i_start,j_start,0] = 0.0
                
    return out_map

# TODO: Experimentally adding --Gustav
def code_parts_support_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold,
                    np.ndarray[ndim=2,dtype=UINT_t] M,
                    outer_frame = 0):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    M : ndarray[ndim=2]
        Mask over the patches that determines which X-Y locations will be used for computing whether the number of edges reaches a certain threshold. Should be a binary array
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef DTYPE_t NINF = -np.inf
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    cdef UINT_t[:,:,:] X_mv = X
    cdef DTYPE_t[:,:,:,:] log_parts_mv = log_parts
    cdef DTYPE_t[:,:,:,:] log_invparts_mv = log_invparts
    cdef UINT_t[:,:] M_mv = M
    cdef DTYPE_t[:,:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=UINT32_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=UINT32)
    cdef UINT32_t[:,:] integral_counts = _integral_counts

    #cdef UINT_t[:,:]
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    with nogil:
        # Build integral image of edge counts
        # First, fill it with edge counts and accmulate across
        # one axis.
        for i in range(X_x_dim):
            for j in range(X_y_dim):
                count = 0
                for z in range(X_z_dim):
                    count += X_mv[i,j,z]
                integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
        # Now accumulate the other axis
        for j in range(X_y_dim):
            for i in range(X_x_dim):
                integral_counts[1+i,1+j] += integral_counts[i,1+j]



        # Code parts
        for i_start in range(X_x_dim-part_x_dim+1):
            i_end = i_start + part_x_dim
            for j_start in range(X_y_dim-part_y_dim+1):
                j_end = j_start + part_y_dim
                #count = _count_edges(X_mv,i_start+i_frame,i_end-i_frame,j_start+i_frame,j_end-i_frame,X_z_dim)

                # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
                cx0 = i_start+i_frame
                cx1 = 1+i_end-i_frame
                cy0 = j_start+i_frame
                cy1 = 1+j_end-i_frame
                count = integral_counts[cx1, cy1] - \
                        integral_counts[cx0, cy1] - \
                        integral_counts[cx1, cy0] + \
                        integral_counts[cx0, cy0]
            
                if count >= threshold:
                    out_map_mv[i_start,j_start] = 0.0
                    out_map_mv[i_start,j_start,0] = NINF 
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            if M_mv[i_start+i,j_start+j]:
                                for z in range(X_z_dim):
                                    # Only if it's inside the support
                                    if X_mv[i_start+i,j_start+j,z]:
                                        for k in range(num_parts):
                                            out_map_mv[i_start,j_start,k+1] += log_parts_mv[k,i,j,z]
                                    else:
                                        for k in range(num_parts):
                                            out_map_mv[i_start,j_start,k+1] += log_invparts_mv[k,i,j,z]
                else:
                    out_map_mv[i_start,j_start,0] = 0.0
                
    return out_map
