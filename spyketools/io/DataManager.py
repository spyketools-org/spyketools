# -*- coding: utf-8 -*-
# @Author: bsotomayorg
# @Date:   2023-01-02 11:39:55
# @Last Modified by:   bsotomayorg
# @Last Modified time: 2023-11-18 17:30:21


import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

class DataManager():
	# class DataManager():
	"""
	`DataManager` helps users to have access to specific data from NWB datasets by applying filters
	across neuron (i.e., brain area) and/or trials/epochs (i.e., stimulus name). 

	Parameters
	----------
	db_path : str
		Full path to dataset (NWB format only).
	verbose : bool
		If True, DataManager show debug information in console.

	**Example:**
	```python
	from spyketools.io.DataManager import DataManager
	db_path = '~/sotomayorb/AllenSDK_datasets/1026124216'

	# reading data
	my_db = DataManager(db_path = db_path, verbose=True)
	my_db.read()

	# selection of trials
	stim_mask   = my_db.apply_epoch_filter_mask({
		'stimulus_name' : 'drifting_gratings',
		'orientation' : [0,45,90,135]
		})
	# selection of neurons per brain area
	neuron_mask = my_db.apply_neuron_filter_mask({
		'areas' : ['VISp', 'VISal','VISpm','VISrl','VISl']
		})

	# data filtering (In-Memory dataset)
	spike_times, ii_spike_times, stim_label = my_db.get_selected_data(
		window_length=0.1,
		stim_label_name='orientation'
	)
	```
	"""

	def __init__(self, db_path, verbose=False):		
		if verbose:
			print ("Setting path = %s" % db_path)
		
		self.session                = None 
		self.stimulus_presentations = None
		self.neurons                = None
		
		# reading metadata in constructor
		self.meta_neuron_areas      = []
		
		# filters
		self.neurons_filter         = {} #NeuronsFilter()
		self.stimuli_filter         = {} #StimuliFilter()
		self.epoch_filter_mask      = None
		self.epoch_neuron_mask      = None
		self.db_path                = db_path
	
	def read(self, verbose=False):
		self.session = EcephysSession.from_nwb_path(self.db_path)
		self.meta_neuron_areas = list(self.session.metadata['structure_acronyms'])
		
		if verbose:
			print ("Session loaded: `%s`" % self.db_path)
		
		# STIM filtering
		self.stimulus_presentations = self.session.stimulus_presentations
		if ('stimulus_name' in self.stimuli_filter):
			self.apply_stimuli_filter(self.stimuli_filter['stimulus_name'])
			if verbose:
				print ("self. stimulus_presentations: Finished")
		else:
			self.stimulus_presentations = self.session.stimulus_presentations
		
		# NEURON filtering
		self.neurons = self.session.units
		if 'area' in self.neurons_filter:
			self.apply_neuron_filter()
			if verbose:
				self.neurons_filter.summary()
				print ("self.neurons: Finished")
		
	def set_neurons_filter(self, neuron_filter):
		if 'area' in neuron_filter:
			self.neurons_filter = neuron_filter['area']
	def set_stimuli_filter(self, stimulus_filter):
		if 'stimulus_name' in stimulus_filter: #if type(stimulus_name)!=type(None):
			self.stimuli_filter = stimulus_filter['stimulus_name']
			
	def apply_neuron_filter(self):
		self.neurons = self.session.units[self.session.units.ecephys_structure_acronym.isin(self.neurons_filter)]
	
	def apply_stimuli_filter(self):
		num_rows  = len(self.session.stimulus_presentations.stimulus_name)
		mask_name = np.zeros(shape=(num_rows), dtype = np.bool)
		for stim_name in self.stimuli_filter.stimulus_name: # for each stimulus name, we add ones to the `mask_name` variable
			mask_name |= (self.stimulus_presentations.stimulus_name == stim_name)
		self.stimulus_presentations = self.stimulus_presentations[mask_name]


	# ---

	#docable
	def apply_epoch_filter_mask(self, dict_filter):
		mask = np.ones(shape=len(self.session.stimulus_presentations), dtype=np.bool)
		for k, v in dict_filter.items():
			if isinstance(v, list):
				temp_mask = np.zeros(shape=len(self.session.stimulus_presentations), dtype=np.bool)
				for item in v:
					temp_mask = temp_mask | (self.session.stimulus_presentations[k] == item)
				mask = mask & temp_mask
			else:
				mask = mask & (self.session.stimulus_presentations[k] == v)
		print ('%i neurons out of %i' % (np.sum(mask),len(mask)))
		self.epoch_filter_mask = np.array(mask)

	#docable
	def apply_neuron_filter_mask(self, dict_filter):
		mask = self.session.units.ecephys_structure_acronym.isin(dict_filter['areas'])
		print ('%i neurons out of %i' % (np.sum(mask),len(mask)))
		self.neuron_filter_mask = np.array(mask)


	# Not Implemented yet # 
	def valid_criterias_epoch_selection(self, dict_filter):
		if not isinstance(dict_filter, dict):
			raise TypeError("Input must be a dictionary (%s found)" % str(type(dict_filter)))
		
		dataset_column_names = list(self.session.stimulus_presentations.keys())
		
		# find if keys are part can be found
		for k in dict_filter.keys():
			if not k in dataset_column_names:
				raise NotImplementedError("dataset does not contain column '%s'." % k)
				
		return True

	#docable
	def get_selected_data(self, window_length, stim_label_name):
		"""
		Method to apply filters and extract data from the public available datasets of
		with NWB data format. 
		Spikes are stored as relative spike times.

		For more details, see [http://help.brain-map.org/display/observatory/Documentation](http://help.brain-map.org/display/observatory/Documentation). 

		Parameters
		----------
		window_length : float
			Window length for spike train analysis.
		stim_label_name : str
			If True, it returns the stim. labels (i.e., drifting gratings' orientation).
		
		Returns
		-------
		spike_times : numpy.ndarray
			Array with relative spike times.
		ii_spikes_times: numpy.ndarray
			`(M,N,2)`-Array with indices per neuron (N) and epoch (M).
		stim_labels: numpy.ndarray
			`M`-dimensional array with stimulus information (e.g. drifting gratings' orientations). 
		"""

		l_start_time = np.array(self.session.stimulus_presentations[self.epoch_filter_mask]['start_time'])
		stim_labels = np.array(self.session.stimulus_presentations[self.epoch_filter_mask][stim_label_name])
		
		sel_neuron_activity = np.array(np.array(list(self.session.spike_times.values())))[self.neuron_filter_mask]
		
		M = len(l_start_time)
		N = np.sum(self.neuron_filter_mask, dtype=np.int)
		
		index = 0
		spike_times = []; ii_spike_times = []
		for i_e in range(M):
			start_time = l_start_time[i_e]
			
			temp_ii_spike_times = []
			for neuron_id in range(N):
				spike_train = sel_neuron_activity[neuron_id]
				st = spike_train[(spike_train >= start_time) & (spike_train < (start_time+window_length))].copy()
				st -= start_time
				
				temp_ii_spike_times.append([index, index+len(st)])
				index += len(st)
				spike_times.append(st)
				
			ii_spike_times.append(temp_ii_spike_times)
			
		spike_times = np.concatenate(spike_times)
		ii_spike_times = np.array(ii_spike_times)
		
		return spike_times, ii_spike_times, stim_labels