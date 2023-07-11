from numba import njit, jit, set_num_threads, prange
import numba
import numpy as np
import itertools
#from ..utils import deprecated

# ==== Distances between two spike trains ==== #

#docable
def ISI_distance(t1, t2, t_start , t_end=0, mode='njit'):
    """
    Computation of ISI[^1] distance between two spike trains with relative spike times.

    *Note:* This method is based on Mariomulansky's implementation (Cython). 
    For further details, see [PySpike github repository](https://github.com/mariomulansky/PySpike/blob/504ded4b3129a1bb7fdcfdc74fb394f838687345/pyspike/cython/cython_distances.pyx).
    
    Parameters
    ----------
    t1 : numpy.array
        First non-empty spike train.
    t2 : numpy.array
        Second non-empty spike train.
    t_start : float
        Start time of spike trains (usually, it equals 0).
    t_end : float
        End time of spike trains (usually, it equals the window length size).

    Returns
    -------
    distance : float
        ISI distance between two spike trains.
    
    [^1]:
        *Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A, Measuring spike train synchrony. J Neurosci Methods 165, 151 (2007)*.
    """
    if t_end == 0 :
        t_end = np.max(spike_times)
    t_start = 0;

    # TODO
    # validations
    # sorted?
    # every spike in corresponding range?
    if mode=='py':
        return py_ISI_distance(t1, t2, t_start, t_end)
    elif mode=='njit':
        return ISI_dist(t1, t2, t_start, t_end)
    else:
        raise NotImplementedError("mode=='%s' is not implemented yet." % mode)

#docable
def ISI_pairwise_distances(spike_times, ii_spike_times, window_length, diag_value=0, num_threads=-1):
    """
    Compute ISI distance directly on spike times of all `M` trials/epochs with `N` channels/neurons
    using all available CPU cores.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times
    ii_spike_times : numpy.ndarray
        `(M,N,2)` dimensional matrix containing the start and end index for the `spike_times` array
        for any given epoch and channel combination.
    window_length : float
        Window length.
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
    from spyketools.distances.ISI import ISI_pairwise_distance
    
    # reading example data
    spike_times    = np.load("demo_dataset_allen/spike_times.npy")
    ii_spike_times = np.load("demo_dataset_allen/ii_spike_times.npy")

    # computation of pairwise distances
    ISI_pairwise_distances(
        spike_times, 
        ii_spike_times, 
        window_length=1.0, 
        diag_value=0)
    ```
    """

    # combination of epochs
    n_epochs = ii_spike_times.shape[0]

    epoch_index_pairs = np.array(
        list(itertools.combinations(range(n_epochs), 2)),
        dtype=int)

    set_nthreads(num_threads)

    return ISI_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, window_length)

@njit
def ISI_dist(s1, s2, t_start, t_end):
    nu1 = 0.; nu2 = 0.
    isi_value = 0.0
    N1 = len(s1); N2 = len(s2)
    if s1[0] > t_start:
        if N1 > 1:
            nu1 = max([s1[0]-t_start, s1[1]-s1[0]]) 
        else:
            nu1 = s1[0]-t_start
        
        index1 = -1
    else:
        if N1 > 1:
            nu1 = s1[1]-s1[0]
        else:
            nu1 = t_end-s1[0]
        index1 = 0

    if s2[0] > t_start:
        if N2 > 1 :
            nu2 = max([s2[0]-t_start, s2[1]-s2[0]]) 
        else:
            nu2 = s2[0]-t_start
        #if N2 > 1 else 
        index2 = -1
    else:
        if N2 > 1:
            nu2 = s2[1]-s2[0]
        else:
            n2 = t_end-s2[0]
        #nu2 = s2[1]-s2[0] if N2 > 1 else t_end-s2[0]
        index2 = 0
    
    last_t = t_start
    curr_isi = abs(nu1-nu2)/max([nu1, nu2])
    index = 1

    while index1+index2 < N1+N2-2:
        if (index1 < N1-1) and ((index2 == N2-1) or
                                (s1[index1+1] < s2[index2+1])):
            index1 += 1
            curr_t = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                if N1 > 1:
                    nu1 = max([t_end-s1[index1], nu1])
                else:
                    nu1 = t_end-s1[index1]
                #nu1 = np.max([t_end-s1[index1], nu1]) if N1 > 1 \
                #      else t_end-s1[index1]
        elif (index2 < N2-1) and ((index1 == N1-1) or
                                  (s1[index1+1] > s2[index2+1])):
            index2 += 1
            curr_t = s2[index2]
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                # edge correction for the end as above
                if N2 > 1:
                    nu2 = max([t_end-s2[index2], nu2])
                else:
                    nu2 = t_end-s2[index2]
        else: # s1[index1+1] == s2[index2+1]
            index1 += 1
            index2 += 1
            curr_t = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                # edge correction for the end as above
                if N1 > 1:
                    nu1 = max([t_end-s1[index1], nu1])
                else:
                    #nu1 = np.max([t_end-s1[index1], nu1]) if N1 > 1 \
                    #  else 
                    t_end-s1[index1]
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                # edge correction for the end as above
                if N2 > 1:
                    nu2 = max([t_end-s2[index2], nu2]) 
                else:
                    nu2 = t_end-s2[index2]
                #nu2 = np.max([t_end-s2[index2], nu2]) if N2 > 1 \
                #      else t_end-s2[index2]
        # compute the corresponding isi-distance
        isi_value += curr_isi * (curr_t - last_t)
        curr_isi = abs(nu1 - nu2) / max([nu1, nu2])
        last_t = curr_t
        index += 1

    isi_value += curr_isi * (t_end - last_t)

    return np.abs(isi_value) / (t_end-t_start)

#@deprecated("This method is only used for testing purposes. Use ISI_distance instead.")
def py_ISI_distance(s1, s2, t_start, t_end):   
    nu1 = 0.; nu2 = 0.
    isi_value = 0.0
    N1 = len(s1); N2 = len(s2)
    if s1[0] > t_start:
        if N1 > 1:
            nu1 = max([s1[0]-t_start, s1[1]-s1[0]]) 
        else:
            nu1 = s1[0]-t_start
        
        index1 = -1
    else:
        if N1 > 1:
            nu1 = s1[1]-s1[0]
        else:
            nu1 = t_end-s1[0]
        index1 = 0

    if s2[0] > t_start:
        if N2 > 1 :
            nu2 = max([s2[0]-t_start, s2[1]-s2[0]]) 
        else:
            nu2 = s2[0]-t_start
        index2 = -1
    else:
        if N2 > 1:
            nu2 = s2[1]-s2[0]
        else:
            n2 = t_end-s2[0]
        index2 = 0
    
    last_t = t_start
    curr_isi = abs(nu1-nu2)/max([nu1, nu2])
    index = 1

    while index1+index2 < N1+N2-2:
        if (index1 < N1-1) and ((index2 == N2-1) or
                                (s1[index1+1] < s2[index2+1])):
            index1 += 1
            curr_t = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                if N1 > 1:
                    nu1 = max([t_end-s1[index1], nu1])
                else:
                    nu1 = t_end-s1[index1]
        elif (index2 < N2-1) and ((index1 == N1-1) or
                                  (s1[index1+1] > s2[index2+1])):
            index2 += 1
            curr_t = s2[index2]
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                if N2 > 1:
                    nu2 = max([t_end-s2[index2], nu2])
                else:
                    nu2 = t_end-s2[index2]
        else:
            index1 += 1
            index2 += 1
            curr_t = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                if N1 > 1:
                    nu1 = max([t_end-s1[index1], nu1])
                else:
                    t_end-s1[index1]
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                if N2 > 1:
                    nu2 = max([t_end-s2[index2], nu2]) 
                else:
                    nu2 = t_end-s2[index2]

        isi_value += curr_isi * (curr_t - last_t)
        curr_isi = abs(nu1 - nu2) / max([nu1, nu2])
        last_t = curr_t
        index += 1

    isi_value += curr_isi * (t_end - last_t)

    return np.abs(isi_value) / (t_end-t_start)

@jit(nopython=True)
def ISI_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, window_length, diag_value):

    # Get data dimensions
    n_epochs = ii_spike_times.shape[0]
    n_channels = ii_spike_times.shape[1]
    n_epoch_index_pairs = epoch_index_pairs.shape[0]

    # Initialize distance matrix
    distances = np.full((n_epochs, n_epochs), np.nan)

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

                neuron_distances[c] = ISI_dist(channel_spikes_e1, channel_spikes_e2, t_start=0., t_end=window_length)

            else:
                nan_count = nan_count + 1

        # Save average and std of distances
        distances[e1, e2] = np.nanmean(neuron_distances)
        distances[e2, e1] = distances[e1, e2]

    percent_nan = nan_count / (n_channels*n_epoch_index_pairs)

    # Set diagonal to zero
    #np.fill_diagonal(distances, diag_value)
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