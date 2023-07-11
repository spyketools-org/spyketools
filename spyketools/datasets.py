import numpy as np
def load_allen_brain_ds(return_labels=True):
    """
    Method to load pre-computed arrays from the public available datasets of 
    Allen Brain Institute through AllenSDK.
    It contains data of Neuropixel recordings in mice's visual areas. 
    Spikes are stored as relative spike times for 1000 neurons randomly selected from 32 sessions.

    For more details, see [http://help.brain-map.org/display/observatory/Documentation](http://help.brain-map.org/display/observatory/Documentation). 

    Parameters
    ----------
    return_labels : bool
        If True, it returns the stim. labels (i.e., drifting gratings' orientation).
    
    Returns
    -------
    spike_times : numpy.ndarray
        Array with relative spike times.
    ii_spikes_times: numpy.ndarray
        `(M,N,2)`-Array with indices per neuron (N) and epoch (M).
    stim_labels: numpy.ndarray
        `M`-dimensional array with drifting gratings' orientations. 

    **Examples:**

    ```python    
    import numpy as np
    from spyketools.datasets import load_allen_brain_ds

    # reading data
    spike_times, ii_spike_times, stim_labels = load_allen_brain_ds()

    spike_times.shape
    # Output: 153635

    # number of epochs/trials
    ii_spike_times.shape[0]
    # Output: 200

    # number of neurons
    ii_spike_times.shape[1]
    # Output: 1000

    # number of stim. labels:
    len(np.unique(stim_labels))
    # Output: 4
    ```
    
    """
    spike_times    = np.load("demo_dataset_allen/spike_times.npy")
    ii_spike_times = np.load("demo_dataset_allen/ii_spike_times.npy")
    stim_label     = np.load("demo_dataset_allen/stim_label.npy")

    if return_labels:
        return spike_times, ii_spike_times, stim_label
    return spike_times, ii_spike_times