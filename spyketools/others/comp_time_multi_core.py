import sys
sys.path.append('/mnt/pns/home/sotomayorb/nogit/spyke-tools-dev/')
#sys.path.append('/mnt/pns/home/sotomayorb/git/spyke-tools-dev/spyketools/distances/SPIKE')
#from importlib import import_module
#import_module('/mnt/pns/home/sotomayorb/git/spyke-tools-dev/spyketools/distances/SPIKE/SPIKE_pairwise_distances')
#sys.path.append('/mnt/pns/home/sotomayorb/git/spyke-tools-dev/spyketools/distances/RISPIKE')
#sys.path.append('/mnt/pns/home/sotomayorb/git/spyke-tools-dev/spyketools/distances/victor_purpura')
#sys.path.append('/mnt/pns/home/sotomayorb/git/spyke-tools-dev/spyketools/distances/spikeship')

from time import time
import numpy as np

from spyketools.distances.SPIKE import SPIKE_pairwise_distances
from spyketools.distances.RISPIKE import RISPIKE_pairwise_distances
from spyketools.distances.victor_purpura import VP_pairwise_distances
from spyketools.distances.spikeship import spikeship_pairwise_distances

##print ("[DBUG] - Successfull Importing!")

# import SPIKE_pairwise_distances
# import RISPIKE_pairwise_distances
# import VP_pairwise_distances
# import spikeship_pairwise_distances

def generate_sim_data(M = 1_000, N = 100, n = 20):
    sim_spike_times = []; sim_ii_spike_times = []; 
    index = 0
    for i_e in range(M):
        temp_ii_spike_times = [];
        for i_N in range(N):
            temp_st = np.random.randint(10_000, size=(n)) / 10_000. 
            temp_st.sort()
            sim_spike_times.append(temp_st)
            temp_ii_spike_times.append([index, index+len(temp_st)])
            index += len(temp_st)
        sim_ii_spike_times.append(temp_ii_spike_times)

    sim_spike_times = np.concatenate(sim_spike_times)
    sim_ii_spike_times = np.array(sim_ii_spike_times)
    return sim_spike_times, sim_ii_spike_times

def main():
    n_rep = 10;
    l_times_py_SPIKE_pw     = []; l_times_njit_SPIKE_pw     = []; 
    l_times_py_RISPIKE_pw   = []; l_times_njit_RISPIKE_pw   = [];  
    l_times_py_VP_pw        = []; l_times_njit_VP_pw        = [];  
    l_times_py_SpikeShip_pw = []; l_times_njit_SpikeShip_pw = []; 

    sim_spike_times, sim_ii_spike_times = generate_sim_data()
    SPIKE_pairwise_distances(sim_spike_times, sim_ii_spike_times[:,:10,:], window_length=1, num_threads=1)
    RISPIKE_pairwise_distances(sim_spike_times, sim_ii_spike_times[:,:10,:], window_length=1, num_threads=1)
    spikeship_pairwise_distances(sim_spike_times, sim_ii_spike_times[:,:10,:], num_threads=-1)
    #VP_distance(t1, t2, cost=n, mode='njit')
    #spikeship_distance(t1, t2, mode='njit')

    print ("metric,time,cores,i_rep")
    for num_threads in [1,2,4,8,16,32]:#,64,128]:#range(1, n_cores):
        for i in range(n_rep):
            sim_spike_times, sim_ii_spike_times = generate_sim_data()
            #print ("cores %i rep %i" % (num_threads, i))

            #### SPIKE ####
            delta_time = time()
            SPIKE_pairwise_distances(sim_spike_times, sim_ii_spike_times, window_length=1, num_threads=num_threads)
            delta_time = time() - delta_time
            l_times_py_SPIKE_pw.append(delta_time)

            print ("%s,%s,%i,%i" % ('SPIKE', delta_time, num_threads, i))

            #### RISPIKE ####
            delta_time = time()
            RISPIKE_pairwise_distances(sim_spike_times, sim_ii_spike_times, window_length=1, num_threads=num_threads)
            delta_time = time() - delta_time
            l_times_py_RISPIKE_pw.append(delta_time)

            print ("%s,%s,%i,%i" % ('RISPIKE', delta_time, num_threads, i))

            #     #### VP ####
            #     delta_time = time()
            #     VP_distance(t1, t2, cost=n, mode='py')
            #     delta_time = time() - delta_time
            #     l_times_py_VP.append(delta_time)

            #     delta_time = time()
            #     VP_distance(t1, t2, cost=n, mode='njit')
            #     delta_time = time() - delta_time
            #     l_times_njit_VP.append(delta_time)


            #### SPIKESHIP ####
            delta_time = time()
            spikeship_pairwise_distances(sim_spike_times, sim_ii_spike_times, num_threads=num_threads)
            delta_time = time() - delta_time
            l_times_py_SpikeShip_pw.append(delta_time)
            print ("%s,%s,%i,%i" % ('SpikeShip', delta_time, num_threads, i))


#l_times_py_SPIKE_pw       = np.array(l_times_py_SPIKE_pw)*1_000 # to miliseconds
#l_times_py_RISPIKE_pw     = np.array(l_times_py_RISPIKE_pw)*1_000 # to miliseconds
#l_times_py_VP_pw          = np.array(l_times_py_VP_pw)*1_000 # to miliseconds
#l_times_py_SpikeShip_pw   = np.array(l_times_py_SpikeShip_pw)*1_000 # to miliseconds




# def generate_plot(l_times_pw):
#     temp_mean = [np.mean(l_times_py_SpikeShip_pw[i*n_rep:(i+1)*n_rep]) for i in range(0,len(l_times_py_SpikeShip_pw)//n_rep) ]
#     temp_std  = [np.std( l_times_py_SpikeShip_pw[i*n_rep:(i+1)*n_rep]) for i in range(0,len(l_times_py_SpikeShip_pw)//n_rep) ]
#     temp_mean = np.array(temp_mean)
#     temp_std = np.array(temp_std)
#     fig, axs = plt.subplots(figsize=(3*2,4), facecolor='w', ncols=2, constrained_layout=True)
#     axs[0].plot([0,1,2,3], temp_mean, marker='o', ms=10, label='SpikeShip')
#     axs[0].set_xticks([0,1,2,3])
#     axs[0].set_xticklabels(["$2^0$","$2^1$","$2^2$","$2^3$"], fontsize=14)
#     axs[0].fill_between(
#         [0,1,2,3], 
#         temp_mean-temp_std, 
#         temp_mean+temp_std, 
#         alpha=0.1
#         )
#     axs[1].plot([0,1,2,3], temp_mean[0]/temp_mean, marker='o', ms=10, label='SpikeShip')
#     axs[1].set_xticks([0,1,2,3])
#     axs[1].set_xticklabels(["$2^0$","$2^1$","$2^2$","$2^3$"], fontsize=14)
#     axs[1].fill_between(
#         [0,1,2,3], 
#         (temp_mean[0]/temp_mean)-(temp_std[0]/temp_std), 
#         (temp_mean[0]/temp_mean)+(temp_std[0]/temp_std), 
#         alpha=0.1
#         )
#     axs[1].axhline(1, color='k', ls='--', marker='', lw=0.5, )

#     axs[0].set_xlabel("# Cores", fontsize=12)
#     axs[0].set_ylabel("Time [ms]", fontsize=12)
#     axs[1].set_xlabel("# Cores", fontsize=12)
#     axs[1].set_ylabel("Efficiency", fontsize=12)

#     fig.suptitle("SpikeShip", fontsize=12)

if __name__ == '__main__':
    main()