import numpy as np
from numba import njit, jit, set_num_threads, prange
import numba
import math
from functools import reduce
import itertools

from numba import int8, int16,int32, int64, float32, vectorize, cuda, njit

# --- General Tools --- #

#docable
def spikeship_distance(t1, t2, mode='njit'):
    """
    Computation of SpikeShip[^1] between two spike trains (relative spike times).
    
    Parameters
    ----------
    t1 : numpy.array
        First non-empty spike train
    t2 : numpy.array
        Second non-empty spike train
    mode : str
        Mode of execution. Options are 'py' (python) and 'njit' (compiled) only.

    Returns
    -------
    f : float
        SpikeShip value for two spike trains

    **Examples:**

    ```python    
    import numpy as np
    from spyketools.distances.spikeship import spikeship_distance

    spike_train_i = np.array([1.2,2,3.2])
    spike_train_j = np.array([1,2,3])

    spikeship_distance(spike_train_i, spike_train_j) 
    ```

    [^1]:
        *Sotomayor-Gómez, B., Battaglia, F. P., & Vinck, M. (2020). SpikeShip: A method for fast, unsupervised discovery of high-dimensional neural spiking patterns. bioRxiv, 2020-06.*
    
    """
    if mode=='njit':
        len_t1 = len(t1)
        len_t2 = len(t2)
        if len_t1 != len_t2:
            if len_t1 > len_t2:
                c_temp = signature_emd_ss(t1, t2)
                C = c_temp[0]
                W = c_temp[1] / (len_t1 * len_t2)
            else:
                c_temp = signature_emd_ss(t2, t1)
                C = -c_temp[0] 
                W =  c_temp[1] / (len_t1 * len_t2)
        else:
            c_temp = signature_emd_ss(t2, t1)
            
            C = c_temp[0]
            W = 1.0/c_temp[1]

        g = weighted_median_(C, W)

        return np.sum(np.abs((C-g)*W))
    else:
        len_t1 = len(t1)
        len_t2 = len(t2)
        if len_t1 != len_t2:
            if len_t1 > len_t2:
                c_temp = py_signature_emd_ss(t1, t2)
                C = c_temp[0]
                W = c_temp[1] / (len_t1 * len_t2)
            else:
                c_temp = py_signature_emd_ss(t2, t1)
                C = -c_temp[0] 
                W =  c_temp[1] / (len_t1 * len_t2)
        else:
            c_temp = py_signature_emd_ss(t2, t1)
            
            C = c_temp[0]
            W = 1.0/c_temp[1]

        g = py_weighted_median_(C, W)

        return np.sum(np.abs((C-g)*W))
        
        

#docable
def spikeship_pairwise_distances(spike_times, ii_spike_times, num_threads=-1):
    """
    Compute SpikeShip[^1] distance directly on spike times of all `M` 
    trials/epochs with `N` channels/neurons using all available CPU cores.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times
    ii_spike_times : numpy.ndarray
        MxNx2 dimensional matrix containing the start and end index for the spike_times array
        for any given epoch and channel combination

    Returns
    -------
    f : numpy.ndarray
        `(M,M)` matrix with $f$ distances (neuron-specific shifts) with `numpy.nan` for unknown distances

    **Example:**

    ```python
    # importing modules
    from spyketools.datasets import load_allen_brain_ds
    from spyketools.proc.distances.spikeship import spikeship_pairwise_distances
    
    # reading data
    spike_times, ii_spike_times, _ = load_allen_brain_ds()
    
    # computation of pairwise distances
    spikeship_pairwise_distances(spike_times, ii_spike_times)
    ```

    [^1]:
        *Sotomayor-Gómez, B., Battaglia, F. P., & Vinck, M. (2020). SpikeShip: A method for fast, unsupervised discovery of high-dimensional neural spiking patterns. bioRxiv, 2020-06.*
    """

    if spike_times.ndim != 1: #First argument must be longer than or equally long as second
        raise AttributeError('First argument must be 1-dim but it is %i-dim' % spike_times.ndim)
    if ii_spike_times.ndim != 3:
        raise AttributeError('Second argument must be 3-dim (i.e., MxNx2) but it is %i-dim' % ii_spike_times.ndim)

    n_epochs = ii_spike_times.shape[0]

    epoch_index_pairs = np.array(
        list(itertools.combinations(range(n_epochs), 2)),
        dtype=int)

    set_nthreads(num_threads)
    
    return distances_SpikeShip(ii_spike_times, spike_times, epoch_index_pairs)

@njit(nopython=True, cache=True)
def signature_emd_ss(x, y):
    #
    # Computation of length of spike times between two pair of epochs e1 and e2. 
    # Compiled by Numba JIT.

    # Parameters
    # ----------
    # x : numpy.ndarray
    #     Vector of spike times for the first epoch.
    # y : numpy.ndarray
    #     Vector of spike times for the second epoch.

    # Note: length of x is greater or equal to y.
    
    # Returns
    # -------
    # emd : numpy.ndarray (2, n)
    #     EMD matrix. It contains the L1 distance between spike times (first component), and the weights for each computation (second component).
    #

    Q = len(x)
    R = len(y)

    if Q == 0 or R == 0:
        return np.array([[np.nan]])

    if Q < R:
        raise AttributeError('First argument must be longer than or equally long as second.')

    if Q == R:
        emd = np.zeros((2,Q)) 
        for c in range(Q): 
            emd[0][c] = -(x[c] - y[c]) #bsotomayor aug 17th
            emd[1][c] = Q
        return emd  
    else:

        # Use integers as weights since they are less prome to precision issues when subtracting
        w_x = R # = Q*R/Q
        w_y = Q # = Q*R/R

        n_flow = _n_flows(Q,R)
        emd = np.zeros((2,n_flow))
        #w   = np.zeros(n_flow)
        q = 0
        r = 0

        #for c in range(n_flow):
        c = 0
        while c < n_flow and q < Q:
            if w_x <= w_y:
                cost = (x[q] - y[r])
                w_y -= w_x
                emd[1][c] = w_x
                w_x = R
                q += 1

                if w_y == 0:
                    w_y = Q
                    r += 1
            else:
                cost = (x[q] - y[r])
                w_x -= w_y
                emd[1][c] = w_y
                w_y = Q
                r += 1
                if w_x == 0:
                    w_x = R
                    q += 1


            emd[0][c] = cost
            c += 1

        return emd[:,:c]


def py_signature_emd_ss(x, y):
    #
    # Computation of length of spike times between two pair of epochs e1 and e2. 
    # Compiled by Numba JIT.

    # Parameters
    # ----------
    # x : numpy.ndarray
    #     Vector of spike times for the first epoch.
    # y : numpy.ndarray
    #     Vector of spike times for the second epoch.

    # Note: length of x is greater or equal to y.
    
    # Returns
    # -------
    # emd : numpy.ndarray (2, n)
    #     EMD matrix. It contains the L1 distance between spike times (first component), and the weights for each computation (second component).
    #

    Q = len(x)
    R = len(y)

    if Q == 0 or R == 0:
        return np.array([[np.nan]])

    if Q < R:
        raise AttributeError('First argument must be longer than or equally long as second.')

    if Q == R:
        emd = np.zeros((2,Q)) 
        for c in range(Q): 
            emd[0][c] = -(x[c] - y[c]) #bsotomayor aug 17th
            emd[1][c] = Q
        return emd  
    else:

        # Use integers as weights since they are less prome to precision issues when subtracting
        w_x = R # = Q*R/Q
        w_y = Q # = Q*R/R

        n_flow = _n_flows(Q,R)
        emd = np.zeros((2,n_flow))
        #w   = np.zeros(n_flow)
        q = 0
        r = 0

        #for c in range(n_flow):
        c = 0
        while c < n_flow and q < Q:
            if w_x <= w_y:
                cost = (x[q] - y[r])
                w_y -= w_x
                emd[1][c] = w_x
                w_x = R
                q += 1

                if w_y == 0:
                    w_y = Q
                    r += 1
            else:
                cost = (x[q] - y[r])
                w_x -= w_y
                emd[1][c] = w_y
                w_y = Q
                r += 1
                if w_x == 0:
                    w_x = R
                    q += 1


            emd[0][c] = cost
            c += 1

        return emd[:,:c]

@jit(nopython=True, cache=True)
def get_Flows(ii_spike_times_e1, ii_spike_times_e2, spike_times ):
    # Computation of flow for spike times between two pair of neurons. 
    # Compiled by Numba JIT.

    # Parameters
    # ----------
    # ii_spike_times_e1 : numpy.ndarray
    #     Vector of indexes of spike times sequence of the first epoch
    # ii_spike_times_e2 : numpy.ndarray
    #     Vector of indexes of spike times sequence of the second epoch
    # spike_times : numpy.ndarray
    #     1 dimensional matrix containing all spike times

    # Returns
    # -------
    # C : numpy.ndarray
    #     Distances between two spike times sequences
    # W : numpy.ndarray
    #     Vector of weights. This values are the least common multiple between neurons

    w_array = np.zeros(ii_spike_times_e1.shape[0], dtype=np.int64)#, dtype=int64)
    
    for ii in range(w_array.shape[0]):
        
        if ((ii_spike_times_e1[ii][1]-ii_spike_times_e1[ii][0] == 0) or (ii_spike_times_e2[ii][1]-ii_spike_times_e2[ii][0] == 0)):
            w_array[ii] = 0
        else:
            len_st_e1 = np.int64(ii_spike_times_e1[ii][1]-ii_spike_times_e1[ii][0])
            len_st_e2 = np.int64(ii_spike_times_e2[ii][1]-ii_spike_times_e2[ii][0])
            
            span = len_st_e1
            if len_st_e1 != len_st_e2:
                if len_st_e1 < len_st_e2:
                    span = _n_flows(len_st_e1, len_st_e2)
                else:
                    span = _n_flows(len_st_e2, len_st_e1)

            w_array[ii] = span
    
    
    # N: number of channels or neurons
    N = ii_spike_times_e1.shape[0]
    optimal_length = int(np.sum(w_array))

    C = np.zeros(optimal_length, dtype=np.float64)
    W = np.zeros(optimal_length, dtype=np.float64)

    # Computation of flow for all the pair of neurons in epochs e1 and e2
    i_start = 0
    
    sum_w = np.sum(w_array)

    c_active_neurons = 0
    for ii in range(N):
        span = 0
        if (
            (ii_spike_times_e1[ii,1]-ii_spike_times_e1[ii,0])>0 and 
            (ii_spike_times_e2[ii,1]-ii_spike_times_e2[ii,0])>0 ):

            st1 = spike_times[ii_spike_times_e1[ii,0]:ii_spike_times_e1[ii,1]]
            st2 = spike_times[ii_spike_times_e2[ii,0]:ii_spike_times_e2[ii,1]]
            
            len_st1 = len(st1)
            len_st2 = len(st2)
            if len(st1) != len(st2):
                if len_st1 > len_st2:
                    span = _n_flows(len_st1, len_st2)
                    c_temp = signature_emd_ss(st1, st2)
                    span = len(c_temp[1])
                    C[i_start:(i_start+span)] = c_temp[0]
                    W[i_start:(i_start+span)] = c_temp[1] / (len_st1 * len_st2)
                else:
                    span = _n_flows(len_st2, len_st1)
                    c_temp = signature_emd_ss(st2, st1)
                    span = len(c_temp[1])
                    C[i_start:(i_start+span)] = -c_temp[0] 
                    W[i_start:(i_start+span)] =  c_temp[1] / (len_st1 * len_st2)
            else:
                span = len_st1 # span equals 'n' when n1 == n2
                w = span
                
                c_temp = signature_emd_ss(st2, st1)
                span = len(c_temp[1])
                
                C[i_start:(i_start+span)] = c_temp[0]
                W[i_start:(i_start+span)] = 1.0/c_temp[1]
            
            # increase the number of active neurons between two epochs
            c_active_neurons += 1
            
        i_start += span
    C = C[:i_start]
    W = W[:i_start]

    # computation of D
    if _areAllOnes(w_array): # single spike patterns
        g = np.median(C)
    else:
        if np.sum(W)>0:     # multi-spike patterns
            g = weighted_median_(x=C,w=W)
        else: # it ocurrs when two epochs do not have flow because one of the is spikeless
            g = np.nan #0.

    return (C - g), W, g, c_active_neurons

@jit(nopython=True, cache=True)
def _areAllOnes(arr):
    for idx in range(len(arr)):
        if (arr[idx] != 1):
            return False
    return True

@jit(nopython=True, cache=True, parallel=True)
def distances_SpikeShip(ii_spike_times, spike_times, epoch_index_pairs):
    n_epochs   = ii_spike_times.shape[0]
    n_neurons   = ii_spike_times.shape[1]
    n_epoch_index_pairs = epoch_index_pairs.shape[0]

    ss_distances = np.full((n_epochs, n_epochs), np.nan)

    for i in prange(n_epoch_index_pairs): # O(M^2)
        e1 = epoch_index_pairs[i,0]
        e2 = epoch_index_pairs[i,1]

        # compute F vector      
        F, W, g, c_active_neurons = get_Flows( # O(N)
            ii_spike_times_e1 = ii_spike_times[e1],
            ii_spike_times_e2 = ii_spike_times[e2],
            spike_times       = spike_times
            )   
        
        if F.shape[0]==1 and g==0: # no shifts
            ss_distances[e1][e2] = np.nan
            ss_distances[e2][e1] = np.nan
        else: # normal case
            ss_distances[e1][e2] = SpikeShip_value(F=F, W=W, N=c_active_neurons) # O(N)
            ss_distances[e2][e1] = ss_distances[e1][e2]
        
    np.fill_diagonal(ss_distances, 0)

    return ss_distances

@jit(int32(int32, int32))
def _n_flows(X,Y):
    return int(X+Y-1)


# --- Numba utils --- #

def set_nthreads(num_threads):
    if num_threads != -1:
        if num_threads > numba.config.NUMBA_DEFAULT_NUM_THREADS:
            raise AttributeError('The number of threads exceeds the number of cores.')
        numba.set_num_threads(num_threads)

def get_num_threads():
    return numba.get_num_threads()


# --- SpikeShip Normalization --- #

@jit(nopython=True, cache=True)
def SpikeShip_value(F, W, N):
    # Computation of Fast Spike Pattern Optimal Trasport Distance.
    # Compiled by Numba JIT.

    # Parameters
    # ----------
    # F : numpy.ndarray
    #     Vector of ss_distances.
    
    # Returns
    # -------
    # SpikeShip normalized value: numpy.float
    #     The SPOTDis approximation using the median as the optimal global shift between
    #     two epochs.
    
    if N<=1:
        return np.nan
    return (((7.)/(10.* (N-1.)))*np.sum(np.absolute(W*F))) # (single spike case)



# ======================================= TOOLS ======================================= #

# Methods adapted from https://github.com/FilippoBovo/robustats/blob/master/c/robustats.c 
# to compute the weighted median in O(N)

@jit(nopython=True)
def swap_2d(x, n2, i, j):
    for k in range(0, n2):
        temp = x[i][k];
        x[i][k] = x[j][k];
        x[j][k] = temp;
    return x

@jit(nopython=True)
def partition_on_kth_element_2d(x, begin, end, n2,  m, k):
    value = x[k][m]

    x = swap_2d(x, n2, k, end);

    i = begin
    for j in range(begin, end):
        
        if (x[j][m] < value):
            x = swap_2d(x, n2, i, j);
            i += 1

    x= swap_2d(x, n2, i, end);

    return i;

@jit(nopython=True)
def partition_on_kth_smallest_2d(x, begin, end, n2, m, k):
    while (True):
        
        if (begin == end):
            return x[begin][m]
        
        pivot_index = begin + int(np.random.random()* (end - begin + 1)); #random_range(begin, end);
        pivot_index = partition_on_kth_element_2d(x, begin, end, n2, m, pivot_index);
        
        if (k == pivot_index):
            return x[k][m];
        elif (k < pivot_index):
            end = pivot_index - 1;
        else:
            begin = pivot_index + 1;


@jit(nopython=True)
def zip_(array_a, array_b, n):
    return np.column_stack((array_a,array_b))

@njit()
def weighted_median(x, w):
    begin = 0; end = len(x) - 1;
    xw_n = end - begin + 1

    xw = zip_(x, w, xw_n)
    w_sum = np.sum(w[:xw_n])

    median, w_middle = 0.0, 0.0;
    w_lower_sum, w_lower_sum_norm, w_higher_sum, w_higher_sum_norm = 0.0, 0.0, 0.0, 0.0
    while(True):
        n = end - begin + 1
        if (n == 1):
            return x[begin]
        elif (n == 2):
            if (w[begin] >= w[end]):
                return x[begin];
            else:
                return x[end];
        else:
            median_index = begin + (n - 1) // 2; # lower median index
            median = partition_on_kth_smallest_2d( xw, begin, end, 2, 0, median_index)

            w_middle = xw[median_index][1];

            w_lower_sum = 0.;
            for i in range(begin, median_index):
                w_lower_sum += xw[i][1];
            w_lower_sum_norm = w_lower_sum / w_sum


            w_higher_sum = 0.;
            for i in range(median_index+1, end+1):
                w_higher_sum += xw[i][1];
            w_higher_sum_norm = w_higher_sum / w_sum

            if (w_lower_sum_norm < 0.5 and w_higher_sum_norm < 0.5):
                return median;
            elif (w_lower_sum_norm > 0.5):
                xw[median_index][1] = xw[median_index][1] + w_higher_sum;
                end = median_index;
            else:
                xw[median_index][1] = xw[median_index][1] + w_lower_sum;
                begin = median_index;

@jit(nopython=True,cache=True)
def weighted_median_(x,w):
    argsort = np.argsort(x)
    w = w[argsort]; x = x[argsort];

    midpoint = 0.5 * np.sum(w)

    if np.any(w > midpoint):
        return (x[w == np.max(w)])[0]
    else:
        cs_weights = np.cumsum(w)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            return np.mean(x[idx:idx+2])
        else:
            return x[idx+1]

def py_weighted_median_(x,w):
    argsort = np.argsort(x)
    w = w[argsort]; x = x[argsort];

    midpoint = 0.5 * np.sum(w)

    if np.any(w > midpoint):
        return (x[w == np.max(w)])[0]
    else:
        cs_weights = np.cumsum(w)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            return np.mean(x[idx:idx+2])
        else:
            return x[idx+1]  

if __name__ == '__main__':
    cc.compile()