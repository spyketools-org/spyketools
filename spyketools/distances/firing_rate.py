import numpy as np
from numba import njit

#########################################################
# GET SC DISS
#########################################################
# get SC diss
@njit
def get_SC_diss(ii_spike_times):
    # function get_SC_diss(ii_spike_times):
    """Return the collection values.
        Returns:
            The collection values.
        """
    # pairwise
    M = ii_spike_times.shape[0]
    N = ii_spike_times.shape[1]
    sc_dis = np.zeros(shape=(M,M))
    for e0 in range(ii_spike_times.shape[0]):
        for e1 in range(e0+1, ii_spike_times.shape[0]):
            tmp_st1 = ii_spike_times[e0,:,1]-ii_spike_times[e0,:,0]
            tmp_st2 = ii_spike_times[e1,:,1]-ii_spike_times[e1,:,0]
            sc_dis[e0, e1] = np.sqrt(np.sum((tmp_st1-tmp_st2)**2))
            sc_dis[e1, e0] = sc_dis[e0, e1]
    return sc_dis


@njit
def get_SC_diss_z(ii_spike_times):
    # pairwise
    M = ii_spike_times.shape[0]
    N = ii_spike_times.shape[1]
    sc_dis = np.zeros(shape=(M, M))
    mat    = np.zeros(shape=(M, N))
    
    for e0 in range(M):
        mat[e0,:] = ii_spike_times[e0,:,1]-ii_spike_times[e0,:,0]
    
    for i_n in range(N):
        if np.std(mat[:, i_n])>0:
            mat[:, i_n] = (mat[:, i_n] - np.mean(mat[:, i_n])) / np.std(mat[:, i_n])

        
    for e0 in range(M):
        for e1 in range(e0+1, ii_spike_times.shape[0]):
            sc_dis[e0, e1] = np.sqrt(np.sum((mat[e0]-mat[e1])**2))
            sc_dis[e1, e0] = sc_dis[e0, e1]
    return sc_dis


def distance(ii_spike_times, mode='simple', normalized=False, window_length=1, nan_diag=False):
    """
    mode: simple, pairwise.
    """
    dist = None
    if mode=='simple':
        raise("mode '%s' is not implemented yet is not implemented" % mode)
    elif mode=='pairwise':
        if normalized:
            dist = get_SC_diss_z(ii_spike_times)
        else:
            dist = get_SC_diss(ii_spike_times)
    else:
        raise("Error: mode '%s' is not implemented" % mode)

    if nan_diag:
        np.fill_diagonal(dist, np.nan)
    return dist/window_length


def FR_distance_single(t1, t2, window_lenght=1, normalize=False, nan_diag=False, mode='py'):
    
    return (len(t1)-len(t2)) / window_lenght

    #if t_end == 0 :
    #    t_end = np.max(spike_times)
    #t_start = 0;

    # TODO
    # validations
    # sorted?
    # every spike in corresponding range?

    #if mode=='py':
    #    distance(t1, t2, t_start, t_end)
    #elif mode=='compiled' or mode=='njit':
    #    distance(t1, t2, t_start, t_end)
    #else:
    #    raise NotImplementedError("mode=='%s' is not implemented yet." % mode)



def FR_distance_pairwise(t1, t2, window_lenght=1, normalize=False, nan_diag=False, mode='py'):
    
    return (len(t1)-len(t2)) / window_lenght