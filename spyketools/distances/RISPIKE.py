import numpy as np
import itertools
from numba import njit, jit, set_num_threads, prange
import numba

#docable
def RISPIKE_distance(t1, t2, t_start , t_end=0, mode='njit'):
    """
    Computation of Rate-Independent SPIKE (RI-SPIKE)[^1] distance between two spike trains 
    with relative spike times.

    *Note:* This method is based on PySpike's implementation (Python). 
    For further details, see [PySpike github repository](https://github.com/mariomulansky/PySpike/blob/504ded4b3129a1bb7fdcfdc74fb394f838687345/pyspike/cython/cython_distances.pyx).
    
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
        SPIKE distance between two spike trains.

    **Examples:**

    ```python    
    import numpy as np
    from spyketools.distances.SPIKE import SPIKE_distance

    spike_train_i = np.array([1.2,2,3.2])
    spike_train_j = np.array([1,2,3])

    SPIKE_distance(spike_train_i, spike_train_j, t_start=0, t_end=5) 
    # Output: 0.1188
    ```

    ```python
    import numpy as np
    from spyketools.distances.SPIKE import SPIKE_distance

    spike_train_i = np.array([1,2,3,4])
    spike_train_j = np.array([1,2,3,3.5])

    SPIKE_distance(spike_train_i, spike_train_j, t_start=0, t_end=5) 
    # Output: 0.1418
    ```
    
    [^1]:
        *Satuvuori E, Mulansky M, Bozanic N, Malvestio I, Zeldenrust F, Lenk K, and Kreuz T, Measures of spike train synchrony for data with multiple time-scales, J Neurosci Methods 287, 25 (2017) *.

    """
    if t_end == 0 :
        t_end = np.max(spike_times)
    t_start = 0;

    # validations
    # sorted?
    # every spike in corresponding range?
    # 

    if mode=='py':
        return py_RISPIKE_distance(t1, t2, t_start, t_end)
    elif mode=='njit':
        return RISPIKE_dist(t1, t2, t_start, t_end)
    else:
        raise NotImplementedError("mode=='%s' is not implemented yet." % mode)

#docable
def RISPIKE_pairwise_distances(spike_times, ii_spike_times, window_length, diag_value=0, num_threads=-1):
    """
    Compute SPIKE distance directly on spike times of all `M` 
    trials/epochs with `N` channels/neurons using all available CPU cores.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times
    ii_spike_times : numpy.ndarray
        `(M,N,2)` dimensional matrix containing the start and end index for the 
        `spike_times` array for any given epoch and channel combination.
    window_length : float
        Window length for edge corrections.
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
    from spyketools.proc.distances.SPIKE import SPIKE_pairwise_distances
    
    # reading example data
    spike_times    = np.load("demo_dataset_allen/spike_times.npy")
    ii_spike_times = np.load("demo_dataset_allen/ii_spike_times.npy")

    # computation of pairwise distances
    SPIKE_pairwise_distances(
        spike_times, 
        ii_spike_times, 
        cost=1.5)
    ```
    """

    M = ii_spike_times.shape[0]
    N = ii_spike_times.shape[1]

    epoch_index_pairs = np.array(list(itertools.combinations(range(M), 2)), dtype=int)

    set_nthreads(num_threads)

    dist, _ = RISPIKE_distance_pw(
        spike_times, ii_spike_times, epoch_index_pairs, t_end=window_length, diag_value=diag_value)

    return dist


def isi_avrg_python(isi1, isi2):
    # TODO: Move
    return 0.5*(isi1+isi2)*(isi1+isi2)

def get_min_dist(spike_time, spike_train, N, start_index, t_start, t_end):
    # TODO: MOVE.
    # start with the distance to the start time
    d = np.abs(spike_time - t_start)
    if start_index < 0:
        start_index = 0
    while start_index < N:
        d_temp = np.abs(spike_time - spike_train[start_index])
        if d_temp > d:
            return d
        else:
            d = d_temp
        start_index += 1

    # finally, check the distance to end time
    d_temp = np.abs(t_end - spike_time)
    if d_temp > d:
        return d
    else:
        return d_temp

def isi_avrg_rf(isi1, isi2):
    # rate free version
    return (isi1+isi2)

@njit
def njit_isi_avrg_rf(isi1, isi2):
    # rate free version
    return (isi1+isi2)

@njit
def get_min_dist_cython(spike_time, spike_train, N, start_index, t_start, t_end):
    # TODO: MOVE.

    # Returns the minimal distance |spike_time - spike_train[i]| 
    # with i>=start_index.
    # start with the distance to the start time
    d = np.abs(spike_time - t_start)
    if start_index < 0:
        start_index = 0
    while start_index < N:
        d_temp = np.abs(spike_time - spike_train[start_index])
        if d_temp > d:
            return d
        else:
            d = d_temp
        start_index += 1

    # finally, check the distance to end time
    d_temp = np.abs(t_end - spike_time)
    if d_temp > d:
        return d
    else:
        return d_temp

@njit
def njit_get_min_dist(spike_time, spike_train, N, start_index, t_start, t_end):
    # Returns the minimal distance |spike_time - spike_train[i]| 
    # with i>=start_index.
    # start with the distance to the start time
    d = np.abs(spike_time - t_start)
    if start_index < 0:
        start_index = 0
    while start_index < N:
        d_temp = np.abs(spike_time - spike_train[start_index])
        if d_temp > d:
            return d
        else:
            d = d_temp
        start_index += 1

    # finally, check the distance to end time
    d_temp = np.abs(t_end - spike_time)
    if d_temp > d:
        return d
    else:
        return d_temp

@njit
def RISPIKE_dist(t1, t2, t_start, t_end):
    spike_value = 0.0

    N1 = len(t1)
    N2 = len(t2)

    t_last = t_start
    t_p1 = t_start
    t_p2 = t_start
    if t1[0] > t_start:
        # dt_p1 = t2[0]-t_start
        t_f1 = t1[0]
        dt_f1 = njit_get_min_dist(t_f1, t2, N2, 0, t_start, t_end)
        isi1 = np.max(np.array([t_f1-t_start, t1[1]-t1[0]]))
        dt_p1 = dt_f1
        s1 = dt_p1*(t_f1-t_start)/isi1 if isi1 else 0
        index1 = -1
    else:
        t_f1 = t1[1]
        dt_f1 = njit_get_min_dist(t_f1, t2, N2, 0, t_start, t_end)
        dt_p1 = 0.0
        isi1 = t1[1]-t1[0]
        s1 = dt_p1
        index1 = 0
    if t2[0] > t_start:
        # dt_p1 = t2[0]-t_start
        t_f2 = t2[0]
        dt_f2 = njit_get_min_dist(t_f2, t1, N1, 0, t_start, t_end)
        dt_p2 = dt_f2
        isi2 = np.max(np.array([t_f2-t_start, t2[1]-t2[0]]))
        s2 = dt_p2*(t_f2-t_start)/isi2 if isi2 else 0
        index2 = -1
    else:
        t_f2 = t2[1]
        dt_f2 = njit_get_min_dist(t_f2, t1, N1, 0, t_start, t_end)
        dt_p2 = 0.0
        isi2 = t2[1]-t2[0]
        s2 = dt_p2
        index2 = 0

    # y_start = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
    # alternative definition without second normalization
    if njit_isi_avrg_rf(isi1, isi2) > 0:
        y_start = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
    else:
        y_start = 0
    index = 1

    while index1+index2 < N1+N2-2:
        # print(index, index1, index2)
        if (index1 < N1-1) and (t_f1 < t_f2 or index2 == N2-1):
            index1 += 1
            # first calculate the previous interval end value
            s1 = dt_f1*(t_f1-t_p1) / isi1 if isi1 > 0 else 0
            # the previous time now was the following time before:
            dt_p1 = dt_f1
            t_p1 = t_f1    # t_p1 contains the current time point
            # get the next time
            if index1 < N1-1:
                t_f1 = t1[index1+1]
            else:
                t_f1 = t_end
            t_curr =  t_p1
            s2 = (dt_p2*(t_f2-t_p1) + dt_f2*(t_p1-t_p2)) / isi2 if isi2 > 0 else 0
            # y_end = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            if njit_isi_avrg_rf(isi1, isi2):
                y_end = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
            else:
                y_end = 0

            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)

            # now the next interval start value
            if index1 < N1-1:
                dt_f1 = njit_get_min_dist(t_f1, t2, N2, index2,
                                            t_start, t_end)
                isi1 = t_f1-t_p1
                s1 = dt_p1
            else:
                dt_f1 = dt_p1
                isi1 = np.max(np.array([t_end-t1[N1-1], t1[N1-1]-t1[N1-2]]))
                # s1 needs adjustment due to change of isi1
                s1 = dt_p1*(t_end-t1[N1-1])/isi1 if isi1 > 0 else 9
            # s2 is the same as above, thus we can compute y2 immediately
            # y_start = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            if njit_isi_avrg_rf(isi1, isi2) > 0:
                y_start = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
            else:
                y_start = 0
        elif (index2 < N2-1) and (t_f1 > t_f2 or index1 == N1-1):
            index2 += 1
            # first calculate the previous interval end value
            s2 = dt_f2*(t_f2-t_p2) / isi2 if isi2 > 0 else 0
            # the previous time now was the following time before:
            dt_p2 = dt_f2
            t_p2 = t_f2    # t_p2 contains the current time point
            # get the next time
            if index2 < N2-1:
                t_f2 = t2[index2+1]
            else:
                t_f2 = t_end
            t_curr = t_p2
            s1 = (dt_p1*(t_f1-t_p2) + dt_f1*(t_p2-t_p1)) / isi1
            # y_end = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            if njit_isi_avrg_rf(isi1, isi2) > 0:
                y_end = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
            else:
                y_end = 0

            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)

            # now the next interval start value
            if index2 < N2-1:
                dt_f2 = njit_get_min_dist(t_f2, t1, N1, index1,
                                            t_start, t_end)
                isi2 = t_f2-t_p2
                s2 = dt_p2
            else:
                dt_f2 = dt_p2
                isi2 = np.max(np.array([t_end-t2[N2-1], t2[N2-1]-t2[N2-2]]))
                # s2 needs adjustment due to change of isi2
                s2 = dt_p2*(t_end-t2[N2-1])/isi2
            # s1 is the same as above, thus we can compute y2 immediately
            # y_start = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            if njit_isi_avrg_rf(isi1, isi2)>0:
                y_start = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
            else:
                y_start = 0
        else: # t_f1 == t_f2 - generate only one event
            index1 += 1
            index2 += 1
            t_p1 = t_f1
            t_p2 = t_f2
            dt_p1 = 0.0
            dt_p2 = 0.0
            t_curr = t_f1
            y_end = 0.0
            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)
            y_start = 0.0
            if index1 < N1-1:
                t_f1 = t1[index1+1]
                dt_f1 = njit_get_min_dist(t_f1, t2, N2, index2,
                                            t_start, t_end)
                isi1 = t_f1 - t_p1
            else:
                t_f1 = t_end
                dt_f1 = dt_p1
                isi1 = np.max(np.array([t_end-t1[N1-1], t1[N1-1]-t1[N1-2]]))
            if index2 < N2-1:
                t_f2 = t2[index2+1]
                dt_f2 = njit_get_min_dist(t_f2, t1, N1, index1,
                                            t_start, t_end)
                isi2 = t_f2 - t_p2
            else:
                t_f2 = t_end
                dt_f2 = dt_p2
                isi2 = np.max(np.array([t_end-t2[N2-1], t2[N2-1]-t2[N2-2]]))
        index += 1
        t_last = t_curr

    s1 = dt_f1*(t_end-t1[N1-1])/isi1 if isi1 > 0 else 0
    s2 = dt_f2*(t_end-t2[N2-1])/isi2 if isi2 > 0 else 0

    if njit_isi_avrg_rf(isi1, isi2) > 0:
        y_end = (s1 + s2) / njit_isi_avrg_rf(isi1, isi2)
    else:
        y_end = 0
    spike_value += 0.5*(y_start + y_end) * (t_end - t_last)

    return np.abs(spike_value) / (t_end-t_start)

def py_RISPIKE_distance(t1, t2, t_start, t_end):
    spike_value = 0.0

    N1 = len(t1)
    N2 = len(t2)

    t_last = t_start
    t_p1 = t_start
    t_p2 = t_start
    if t1[0] > t_start:
        # dt_p1 = t2[0]-t_start
        t_f1 = t1[0]
        dt_f1 = get_min_dist_cython(t_f1, t2, N2, 0, t_start, t_end)
        isi1 = np.max([t_f1-t_start, t1[1]-t1[0]])
        dt_p1 = dt_f1
        s1 = dt_p1*(t_f1-t_start)/isi1
        index1 = -1
    else:
        t_f1 = t1[1]
        dt_f1 = get_min_dist_cython(t_f1, t2, N2, 0, t_start, t_end)
        dt_p1 = 0.0
        isi1 = t1[1]-t1[0]
        s1 = dt_p1
        index1 = 0
    if t2[0] > t_start:
        # dt_p1 = t2[0]-t_start
        t_f2 = t2[0]
        dt_f2 = get_min_dist_cython(t_f2, t1, N1, 0, t_start, t_end)
        dt_p2 = dt_f2
        isi2 = np.max([t_f2-t_start, t2[1]-t2[0]])
        s2 = dt_p2*(t_f2-t_start)/isi2
        index2 = -1
    else:
        t_f2 = t2[1]
        dt_f2 = get_min_dist_cython(t_f2, t1, N1, 0, t_start, t_end)
        dt_p2 = 0.0
        isi2 = t2[1]-t2[0]
        s2 = dt_p2
        index2 = 0

    y_start = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0
    index = 1

    while index1+index2 < N1+N2-2:
        if (index1 < N1-1) and (t_f1 < t_f2 or index2 == N2-1):
            index1 += 1
            s1 = dt_f1*(t_f1-t_p1) / isi1 if isi1 > 0 else 0
            dt_p1 = dt_f1
            t_p1 = t_f1
            if index1 < N1-1:
                t_f1 = t1[index1+1]
            else:
                t_f1 = t_end
            t_curr =  t_p1
            s2 = (dt_p2*(t_f2-t_p1) + dt_f2*(t_p1-t_p2)) / isi2 if isi2 > 0 else 0
            y_end = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0

            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)

            if index1 < N1-1:
                dt_f1 = get_min_dist_cython(t_f1, t2, N2, index2,
                                            t_start, t_end)
                isi1 = t_f1-t_p1
                s1 = dt_p1
            else:
                dt_f1 = dt_p1
                isi1 = np.max([t_end-t1[N1-1], t1[N1-1]-t1[N1-2]])
                s1 = dt_p1*(t_end-t1[N1-1])/isi1
            y_start = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0
        elif (index2 < N2-1) and (t_f1 > t_f2 or index1 == N1-1):
            index2 += 1
            # first calculate the previous interval end value
            s2 = dt_f2*(t_f2-t_p2) / isi2 if isi2 > 0 else 0
            # the previous time now was the following time before:
            dt_p2 = dt_f2
            t_p2 = t_f2    # t_p2 contains the current time point
            # get the next time
            if index2 < N2-1:
                t_f2 = t2[index2+1]
            else:
                t_f2 = t_end
            t_curr = t_p2
            s1 = (dt_p1*(t_f1-t_p2) + dt_f1*(t_p2-t_p1)) / isi1 if isi1 > 0 else 0
            # y_end = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            y_end = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0

            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)

            # now the next interval start value
            if index2 < N2-1:
                dt_f2 = get_min_dist_cython(t_f2, t1, N1, index1,
                                            t_start, t_end)
                isi2 = t_f2-t_p2
                s2 = dt_p2
            else:
                dt_f2 = dt_p2
                isi2 = np.max([t_end-t2[N2-1], t2[N2-1]-t2[N2-2]])
                # s2 needs adjustment due to change of isi2
                s2 = dt_p2*(t_end-t2[N2-1])/isi2
            # s1 is the same as above, thus we can compute y2 immediately
            # y_start = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            # alternative definition without second normalization
            y_start = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0

        else: # t_f1 == t_f2 - generate only one event
            index1 += 1
            index2 += 1
            t_p1 = t_f1
            t_p2 = t_f2
            dt_p1 = 0.0
            dt_p2 = 0.0
            t_curr = t_f1
            y_end = 0.0
            spike_value += 0.5*(y_start + y_end) * (t_curr - t_last)
            y_start = 0.0
            if index1 < N1-1:
                t_f1 = t1[index1+1]
                dt_f1 = get_min_dist_cython(t_f1, t2, N2, index2,
                                            t_start, t_end)
                isi1 = t_f1 - t_p1
            else:
                t_f1 = t_end
                dt_f1 = dt_p1
                isi1 = np.max([t_end-t1[N1-1], t1[N1-1]-t1[N1-2]])
            if index2 < N2-1:
                t_f2 = t2[index2+1]
                dt_f2 = get_min_dist_cython(t_f2, t1, N1, index1,
                                            t_start, t_end)
                isi2 = t_f2 - t_p2
            else:
                t_f2 = t_end
                dt_f2 = dt_p2
                isi2 = np.max([t_end-t2[N2-1], t2[N2-1]-t2[N2-2]])
        index += 1
        t_last = t_curr
    s1 = dt_f1*(t_end-t1[N1-1])/isi1 if isi1 > 0 else 0 
    s2 = dt_f2*(t_end-t2[N2-1])/isi2 if isi2 > 0 else 0 
    # alternative definition without second normalization
    y_end = (s1 + s2) / isi_avrg_rf(isi1, isi2) if isi_avrg_rf(isi1, isi2) > 0 else 0

    spike_value += 0.5*(y_start + y_end) * (t_end - t_last)
    
    return np.abs(spike_value) / (t_end-t_start)


@jit(nopython=True, cache=True, parallel=True)
def RISPIKE_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, t_end, diag_value):
    if t_end == 0 :
        t_end = np.max(spike_times)
    t_start = 0;

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
            if ((ii_spike_times[e1,c,1] - ii_spike_times[e1,c,0]) > 1
                and (ii_spike_times[e2,c,1] - ii_spike_times[e2,c,0]) > 1):

                channel_spikes_e1 = spike_times[ii_spike_times[e1,c,0]:ii_spike_times[e1,c,1]]
                channel_spikes_e2 = spike_times[ii_spike_times[e2,c,0]:ii_spike_times[e2,c,1]]

                # EMD
                #print ("st1", channel_spikes_e1)
                #print ("st2", channel_spikes_e2)
                #if len(channel_spikes_e1) >= len(channel_spikes_e2):
                neuron_distances[c] = np.abs(RISPIKE_dist(channel_spikes_e1, channel_spikes_e2, t_start, t_end))
                #else:
                #    neuron_distances[c] = spike_distance_rf_cython(channel_spikes_e2, channel_spikes_e1, t_start, t_end)

            else:
                nan_count = nan_count + 1

        # Save average and std of distances
        distances[e1, e2] = np.nanmean(neuron_distances)
        distances[e2, e1] = distances[e1, e2]

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