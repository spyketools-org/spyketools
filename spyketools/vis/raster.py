import matplotlib.pyplot as plt
import numpy as np

def plot_dissimilarity_matrix(diss_matrix, title="", xlabel="", ylabel="", colorbar_label="", fill_value=0, figsize=(5,4), ax=None, cmap='viridis', show_colorbar=True, figpath=""):
    temp = diss_matrix.copy()
    np.fill_diagonal(temp, fill_value)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')
    
    im = ax.imshow(temp, cmap=cmap)
    if show_colorbar:
        if len(colorbar_label)!=0:
            plt.colorbar(im, label=colorbar_label,ax=ax)
        else:
            plt.colorbar(im, ax=ax)
    if len(xlabel)!=0:
        ax.set_xlabel(xlabel)
    if len(ylabel)!=0:
        ax.set_ylabel(ylabel)
    if len(title)!=0:
        ax.set_title(title)
    if len(figpath)!=0:
        plt.savefig(figpath, bbox_inches="tight")
        print("[INFO]\tFigure was saved as '%s'" % figpath)

def plot_raster_spike_trains(spike_times, ii_spike_times, epoch_id, xmin=None, xmax=None, figsize=(5,8), ax=None, title="", xlabel="", ylabel="", figpath=""):
    """Creates a raster plot of Nuerons ID versus time given a epoch_id."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='w')

    for i_N in range(ii_spike_times.shape[1]):
        temp_spike_train = spike_times[ii_spike_times[epoch_id, i_N, 0]:ii_spike_times[epoch_id, i_N, 1]]
        ax.plot(temp_spike_train, np.ones(shape=len(temp_spike_train))*i_N, marker='|', color='k', ls='', ms=3)   
    #    else:
    #    for i_N in range(ii_spike_times.shape[1]):
    #        temp_spike_train = spike_times[ii_spike_times[epoch_id, i_N, 0]:ii_spike_times[epoch_id, i_N, 1]]
    #        ax.plot(temp_spike_train, np.ones(shape=len(temp_spike_train))*i_N, marker='|', color='k', ls='', ms=3) 
    #    ax.
    ax.set_ylim([-1.5, ii_spike_times.shape[1]-0.5])
    if not(xmin is None):
        ax.set_xlim(left=xmin)
    if not(xmax is None):
        ax.set_xlim(right=xmax)

    if len(xlabel)!=0:
        ax.set_xlabel(xlabel)
    if len(ylabel)!=0:
        ax.set_ylabel(ylabel)
    if len(title)!=0:
        ax.set_title(title)
    if len(figpath)!=0:
        plt.savefig(figpath, bbox_inches="tight")
        print("[INFO]\tFigure was saved as '%s'" % figpath)