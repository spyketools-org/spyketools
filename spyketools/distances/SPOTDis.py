from numba import jit, prange
import numpy as np
import itertools


@jit(nopython=True)
def signature_emd_(x, y):
    """
    A fast implementation of the EMD on sparse 1D signatures like described in:
    Grossberger, L., Battaglia, FP. and Vinck, M. (2018). Unsupervised clustering
    of temporal patterns in high-dimensional neuronal ensembles using a novel
    dissimilarity measure. Source: https://github.com/LGro/spot/tree/main.

    Parameters
    ----------
    x : numpy.ndarray
        List of occurrences / a histogram signature
        Note: Needs to be non-empty and longer or equally long as y
    y : numpy.ndarray
        List of occurrences / a histogram signature
        Notes: Needs to be non-empty and shorter or equally long as x

    Returns
    -------
    distance : float
        Earth Mover's Distances between the two signatures / occurrence lists

    """

    Q = len(x)
    R = len(y)

    if Q == 0 or R == 0:
        return np.nan

    if Q < R:
        raise AttributeError('First argument must be longer than or equally long as second.')

    x.sort()
    y.sort()

    # Use integers as weights since they are less prome to precision issues when subtracting
    w_x = R # = Q*R/Q
    w_y = Q # = Q*R/R

    emd = 0.
    q = 0
    r = 0

    while q < Q:
        if w_x <= w_y:
            cost = w_x * abs(x[q] - y[r])
            w_y -= w_x
            w_x = R
            q += 1
        else:
            cost = w_y * abs(x[q] - y[r])
            w_x -= w_y
            w_y = Q
            r += 1

        emd += cost

    # Correct for the initial scaling to integer weights
    return emd/(Q*R)

@jit(nopython=True, parallel=True)
def xcorr_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs):
    """Compute distances between channel cross correlation pairs using all available CPU cores.
    The specific type of distance is provided via the parameter 'metric'.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times

    ii_spike_times : numpy.ndarray
        MxNx2 dimensional matrix containing the start and end index for the spike_times array
        for any given epoch and channel combination

    epoch_index_pairs : numpy.ndarray
        (M*(M-1)/2)x2 dimensional matrix containing all unique epoch index pairs

    Returns
    -------
    distances : numpy.ndarray
        MxM distance matrix with numpy.nan for unknown distances and on the diagonal
    """

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

        # Compute distances for all xcorr pairs between the two epochs
        xcorr_distances = np.full(int(n_channels * (n_channels-1) / 2), np.nan)
        n_xcorr_distances = 0
        i_xcorr_distance = -1
        for c1 in range(n_channels):
            for c2 in range(c1):
                i_xcorr_distance += 1

                # Only compute the xcorrs and distance in case there is a spike in all relevant channels
                if ((ii_spike_times[e1,c1,1] - ii_spike_times[e1,c1,0]) > 0
                    and (ii_spike_times[e1,c2,1] - ii_spike_times[e1,c2,0]) > 0
                    and (ii_spike_times[e2,c1,1] - ii_spike_times[e2,c1,0]) > 0
                    and (ii_spike_times[e2,c2,1] - ii_spike_times[e2,c2,0]) > 0):

                    # Compute the xcorrs
                    xcorr1 = xcorr_list(
                            spike_times[ii_spike_times[e1,c1,0]:ii_spike_times[e1,c1,1]],
                            spike_times[ii_spike_times[e1,c2,0]:ii_spike_times[e1,c2,1]])
                    xcorr2 = xcorr_list(
                            spike_times[ii_spike_times[e2,c1,0]:ii_spike_times[e2,c1,1]],
                            spike_times[ii_spike_times[e2,c2,0]:ii_spike_times[e2,c2,1]])

                    # EMD
                    if len(xcorr1) >= len(xcorr2):
                        xcorr_distances[i_xcorr_distance] = signature_emd_(xcorr1, xcorr2)
                    else:
                        xcorr_distances[i_xcorr_distance] = signature_emd_(xcorr2, xcorr1)

                    n_xcorr_distances = n_xcorr_distances + 1
                else:
                    nan_count = nan_count + 1

        # Save average xcorr distance
        if n_xcorr_distances > 0:
            distances[e1, e2] = np.nanmean(xcorr_distances)
            distances[e2, e1] = distances[e1, e2]

    percent_nan = nan_count / ((n_channels*(n_channels-1)/2)*n_epoch_index_pairs)

    return distances, percent_nan


@jit(nopython=True, parallel=True)
def spike_spotdis_cpu_(spike_times, ii_spike_times, epoch_index_pairs):
    """Compute the given metric directly on spike times of all M epochs with N channels
    using all available CPU cores.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times

    ii_spike_times : numpy.ndarray
        MxNx2 dimensional matrix containing the start and end index for the spike_times array
        for any given epoch and channel combination

    epoch_index_pairs : numpy.ndarray
        (M*(M-1)/2)x2 dimensional matrix containing all unique epoch index pairs

    Returns
    -------
    distances : numpy.ndarray
        MxM distance matrix with numpy.nan for unknown distances and on the diagonal
    """

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

                # EMD
                if len(channel_spikes_e1) >= len(channel_spikes_e2):
                    neuron_distances[c] = signature_emd_(channel_spikes_e1, channel_spikes_e2)
                else:
                    neuron_distances[c] = signature_emd_(channel_spikes_e2, channel_spikes_e1)

            else:
                nan_count = nan_count + 1

        # Save average and std of distances
        distances[e1, e2] = np.nanmean(neuron_distances)
        distances[e2, e1] = distances[e1, e2]

    percent_nan = nan_count / (n_channels*n_epoch_index_pairs)

    # Set diagonal to zero
    for i in range(n_epochs):
        distances[i, i] = 0
        #distance_stds[i, i] = 0

    #return distances, distance_stds, percent_nan
    return distances, percent_nan


def distances(spike_times, ii_spike_times, epoch_length=1.0, metric='SPOTD_xcorr'):
    """Compute temporal distances based on various versions of the SPOTDis, using CPU parallelization.

    Parameters
    ----------
    spike_times : numpy.ndarray
        1 dimensional matrix containing all spike times

    ii_spike_times : numpy.ndarray
        MxNx2 dimensional matrix containing the start and end index for the spike_times array
        for any given epoch and channel combination

    metric : str
        Pick the specific metric by combining the metric ID with either 'xcorr' to compute it on
        pairwise xcorr histograms or 'times' to compute it directly on spike times.
        Currently available:
            * SPOTD_xcorr
            * SPOTD_xcorr_pooled
            * SPOTD_spikes

    Returns
    -------
    distances : numpy.ndarray
        MxM distance matrix with numpy.nan for unknown distances
    """

    n_epochs = ii_spike_times.shape[0]

    epoch_index_pairs = np.array(
        list(itertools.combinations(range(n_epochs), 2)),
        dtype=int)

    # SPOTDis comparing the pairwise xcorrs of channels
    if metric == 'SPOTD_xcorr':
        distances, percent_nan = xcorr_spotdis_cpu_(
            spike_times, ii_spike_times, epoch_index_pairs)
        distances = distances / (2*epoch_length)

    # SPOTDis comparing the xcorr of a channel with all other channels pooled
    elif metric == 'SPOTD_xcorr_pooled':
        distances, percent_nan = xcorr_pooled_spotdis_cpu_(
            spike_times, ii_spike_times, epoch_index_pairs)
        distances = distances / (2*epoch_length)

    # SPOTDis comparing raw spike trains
    elif metric == 'SPOTD_spikes':
        distances, percent_nan = spike_spotdis_cpu_(
            spike_times, ii_spike_times, epoch_index_pairs)
        distances = distances / epoch_length

    # Otherwise, raise exception
    else:
        raise NotImplementedError('Metric "{}" unavailable, check doc-string for alternatives.'.format(
            metric))

    np.fill_diagonal(distances, 0)

    return distances

@jit(nopython=True)
def xcorr_list(in1, in2):
    """List of all time delays from a full cross correlation of the two inputs

    Parameters
    ----------
    in1 : numpy.ndarray
        Occurence times / indices
    in2 : numpy.ndarray
        Occurence times / indices
    """

    n1 = len(in1)
    n2 = len(in2)

    C = [0.0]*(n1*n2)
    for i in range(n1):
        for j in range(n2):
            C[i*n2+j] = in2[j] - in1[i]

    return C