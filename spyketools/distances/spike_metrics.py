import numpy as np
from numba import jit, set_num_threads, prange
import itertools

from .victor_purpura_distance import victor_purpura_distance_pw
from .firing_rate_distance import get_SC_diss, get_SC_diss_z
from .ISI_distance import ISI_distance_pw
from .SPIKE_distance import SPIKE_distance_pw
from .RISPIKE_distance import RISPIKE_distance_pw
from spikeship import spikeship

def pairwise_spike_distance(spike_times, ii_spike_times, metric, attrs={}):

	# attrs management
	# firing rate related
	if (not 'window_length' in attrs) or (attrs['window_length'] is None):
		window_length = np.max(spike_times)
	else:
		window_length = attrs['window_length']
	# vp-related
	if (not 'cost' in attrs) or (attrs['cost'] is None):
		cost = np.mean(ii_spike_times[:,:,1]-ii_spike_times[:,:,0])
	else:
		cost = attrs['cost']

	# eval of metric
	if metric == "firing_rates":
		dist = get_SC_diss(ii_spike_times)/window_length

	elif metric == 'firing_rates_z':
		dist = get_SC_diss_z(ii_spike_times)/window_length

	else:

		if metric.lower() == "spikeship":
			dist = spikeship.distances(spike_times, ii_spike_times)
		else:
			M = ii_spike_times.shape[0]
			N = ii_spike_times.shape[1]
			epoch_index_pairs = np.array(list(itertools.combinations(range(M), 2)), dtype=int)

			if metric == "victor_purpura":
				dist, _ = victor_purpura_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, cost)

			elif metric == "ISI":
				dist, _ = ISI_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, window_length)

			elif metric == "SPIKE":
				dist, _ = SPIKE_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, window_length)

			elif metric == "RI-SPIKE":
				dist, _ = RISPIKE_distance_pw(spike_times, ii_spike_times, epoch_index_pairs, window_length)
			else:
				raise NotImplementedError("metric '%s' is not implemented." % metric)
	return dist

def spike_train_distance(st1, st2, metric, attrs={}):
	pass
