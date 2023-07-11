import numpy as np
from numba import njit, jit, set_num_threads, prange
import numba
import itertools # pairwise

#docable
def VP_distance(t1, t2, cost=None, mode='njit'):
    """
    Computation of Victor-Purpura[^1] distance between two spike trains 
    with relative spike times.

    *Note:* This method is based on Mariomulansky's implementation (Python). 
    For further details, see [Elephant github repository](https://github.com/nicodjimenez/fit_neuron/blob/
        7b18b6599a3324a3418045282818363fea9aede5/fit_neuron/evaluate/
        spkd_lib.py#L178).
    
    Parameters
    ----------
    t1 : numpy.array
        First non-empty spike train.
    t2 : numpy.array
        Second non-empty spike train.
    cost : float
        Cost per unit time to move a spike. If not specified, cost equals the 
        average of lengths of t1 and t2.
    mode : str
        Mode of execution. Options are 'py' (python) and 'njit' (compiled) only.


    Returns
    -------
    distance : float
        Victor-Purpura distance between two spike trains.

    **Examples:**
    ```python
    import numpy as np
    from spyketools.distances.victor_purpura_distance import VP_distance_single

    spike_train_i = np.array([1,2,3,4])
    spike_train_j = np.array([1,2,3])
    cost = 1
    VP_distance_single(spike_train_i, spike_train_j, cost=cost) 
    # Output: 1.0
    ```

    ```python    
    import numpy as np
    from spyketools.distances.victor_purpura_distance import VP_distance_single

    spike_train_i = np.array([1.2,2,3.2])
    spike_train_j = np.array([1,2,3])
    cost = 1
    VP_distance_single(spike_train_i, spike_train_j, cost=cost) 
    # Output: 0.4
    ```

    ```python
    import numpy as np
    from spyketools.distances.victor_purpura_distance import VP_distance_single

    spike_train_i = np.array([1,2,3,4])
    spike_train_j = np.array([1,2,3,3.5])
    cost = 0.5

    VP_distance_single(spike_train_i, spike_train_j, cost=cost) 
    # Output: 0.25
    ```

    [^1]:
        *Aronov, Dmitriy. "Fast algorithm for the metric-space analysis
        of simultaneous responses of multiple single neurons." Journal
        of Neuroscience Methods 124.2 (2003): 175-179.*
        
    """
    if cost is None:
        cost = (len(st1)+len(st2))/2.

    if mode=='py':
        return py_victor_purpura_distance(t1, t2, cost)
    elif mode=='njit':
        return victor_purpura_distance(t1, t2, cost)

#docable
def VP_pairwise_distances(spike_times, ii_spike_times, cost=None, diag_value=0, num_threads=-1):
    """
    Compute Victor-Purpura (VP) distance directly on spike times of all `M` 
    trials/epochs with `N` channels/neurons using all available CPU cores.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times
    ii_spike_times : numpy.ndarray
        `(M,N,2)` dimensional matrix containing the start and end index for the 
        `spike_times` array for any given epoch and channel combination.
    cost : float
        Cost per unit time to move a spike. If not specified, cost equals to the 
        average spike counts.
    diag_value : float
        Value to fill the diagonal.

    Returns
    -------
    distances : numpy.ndarray
        `(M,M)` distance matrix.
    
    **Example:**

    ```python
    # importing modules
    import numpy as np
    from spyketools.distances.victor_purpura import VP_pairwise_distances
    
    # reading example data
    spike_times    = np.load("demo_dataset_allen/spike_times.npy")
    ii_spike_times = np.load("demo_dataset_allen/ii_spike_times.npy")

    # computation of pairwise distances
    VP_pairwise_distances(
        spike_times, 
        ii_spike_times, 
        cost=1.5)
    ```
    
    !!! warning "Runtime warning"
        The computation of this distance might take a long time! 
        The computational complexity of the pairwise VP distances is $\mathcal{O}(M^2N^2n^2)$[^1][^2][^3], 
        where:

        + $M$: # trials
        + $N$: # neurons, and 
        + $n$: average spike counts across trials.

    [^1]:
        *Aronov, Dmitriy. "Fast algorithm for the metric-space analysis
        of simultaneous responses of multiple single neurons." Journal
        of Neuroscience Methods 124.2 (2003): 175-179.*
    [^2]:
        *Victor, J. D. & Purpura, K. P. Nature and precision of temporal coding in 
        visual cortex: a metric-space analysis. J. neurophysiology 76, 1310–1326 (1996).*
    [^3]:
        *Victor, J. D. & Purpura, K. P. Metric-space analysis of spike trains: theory, 
        algorithms and application. Network: computation neural systems 8, 127–164 (1997).*
    """

    M = ii_spike_times.shape[0]
    N = ii_spike_times.shape[1]
    if cost is None:
        cost = np.mean(ii_spike_times[:,:,1]-ii_spike_times[:,:,0])

    epoch_index_pairs = np.array(list(itertools.combinations(range(M), 2)), dtype=int)

    set_nthreads(num_threads)

    dist, _ = victor_purpura_distance_pw(
        spike_times, ii_spike_times, epoch_index_pairs, cost, diag_value)

    return dist

def py_victor_purpura_distance(t1, t2, cost=0.01):
    
    nspi=len(t1)
    nspj=len(t2)

    if cost==0:
        d=abs(nspi-nspj)
        return d
    elif cost==np.Inf:
        d=nspi+nspj;
        return d

    t1.sort()
    t2.sort()

    scr = np.zeros( (nspi+1,nspj+1) )

    # INITIALIZE MARGINS WITH COST OF ADDING A SPIKE

    scr[:,0] = np.arange(0,nspi+1)
    scr[0,:] = np.arange(0,nspj+1)

    if nspi and nspj:
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                scr[i,j] = np.min(np.array([
                    scr[i-1,j]+1,
                    scr[i,j-1]+1,
                    scr[i-1,j-1]+cost*abs(t1[i-1]-t2[j-1])]))

    d=scr[nspi,nspj]
    return d

@jit(nopython=True)
def victor_purpura_distance(t1, t2, cost=0.01):

    nspi=len(t1)
    nspj=len(t2)

    if cost==0:
        d=abs(nspi-nspj)
        return d
    elif cost==np.Inf:
        d=nspi+nspj;
        return d

    t1.sort()
    t2.sort()

    scr = np.zeros( (nspi+1,nspj+1) )

    # INITIALIZE MARGINS WITH COST OF ADDING A SPIKE

    scr[:,0] = np.arange(0,nspi+1)
    scr[0,:] = np.arange(0,nspj+1)

    if nspi and nspj:
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                scr[i,j] = np.min(np.array([
                    scr[i-1,j]+1,
                    scr[i,j-1]+1,
                    scr[i-1,j-1]+cost*abs(t1[i-1]-t2[j-1])]))

    d=scr[nspi,nspj]
    return d

@jit(nopython=True, cache=True, parallel=True)
def victor_purpura_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, cost, diag_value):


    # Get data dimensions
    n_epochs = ii_spike_times.shape[0]
    n_channels = ii_spike_times.shape[1]
    n_epoch_index_pairs = epoch_index_pairs.shape[0]

    # Initialize distance matrix
    distances = np.full((n_epochs, n_epochs), np.nan)
    #distance_stds = np.full((n_epochs, n_epochs), np.nan)

    nan_count = 0.0

    # For each epoch pair
    for i in prange(n_epoch_index_pairs):
        e1 = epoch_index_pairs[i,0]
        e2 = epoch_index_pairs[i,1]

        # Compute distances for all neuron pairs between the two epochs
        neuron_distances = np.full(n_channels, np.nan)
        for c in range(n_channels):
            # Only compute the emd if there is a spike in that channels in both epochs
            if ((ii_spike_times[e1,c,1] - ii_spike_times[e1,c,0]) > 0
                and (ii_spike_times[e2,c,1] - ii_spike_times[e2,c,0]) > 0):

                channel_spikes_e1 = spike_times[ii_spike_times[e1,c,0]:ii_spike_times[e1,c,1]]
                channel_spikes_e2 = spike_times[ii_spike_times[e2,c,0]:ii_spike_times[e2,c,1]]

                neuron_distances[c] = victor_purpura_distance(channel_spikes_e1, channel_spikes_e2, cost)

            else:
                nan_count = nan_count + 1

        # Save average and std of distances
        distances[e1, e2] = np.nanmean(neuron_distances)
        distances[e2, e1] = distances[e1, e2]

        #distance_stds[e1, e2] = np.nanstd(neuron_distances)
        #distance_stds[e2, e1] = distance_stds[e1, e2]

    percent_nan = nan_count / (n_channels*n_epoch_index_pairs)

    # Set diagonal to zero
    for i in range(n_epochs):
        distances[i, i] = diag_value
        #distance_stds[i, i] = 0

    #return distances, distance_stds, percent_nan
    return distances, percent_nan

def set_nthreads(num_threads):
    if num_threads != -1:
        if num_threads > numba.config.NUMBA_DEFAULT_NUM_THREADS:
            raise AttributeError('The number of threads exceeds the number of cores.')
        numba.set_num_threads(num_threads)

def get_num_threads():
    return numba.get_num_threads()