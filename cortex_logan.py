# -*- coding: utf-8 -*-
# Import necessary modules
import brian2 as b2
from brian2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elephant.gpfa import GPFA
import quantities as pq
import neo
from mpl_toolkits import mplot3d

b2.start_scope()  # Ensures any Brian object created before the function is called aren't included in the next run
###############################################################################
###########                       Parameters                        ###########
###############################################################################

d_ex = 1.5 * ms  # Excitatory delay
std_d_ex = 0.75 * ms  # Std. Excitatory delay
d_in = 0.80 * ms  # Inhibitory delay
std_d_in = 0.4 * ms  # Std. Inhibitory delay
tau_syn = 0.5 * ms  # Post-synaptic current time constant
tau_m = 10.0 * ms  # membrane time constant
tau_ref = 2.0 * ms  # absolute refractory period
Cm = 250.0 * pF  # membrane capacity
v_r = -65.0 * mV  # reset potential
v_th = -50.0 * mV  # fixed firing threshold
w_ex = 87.8 * pA  # excitatory synaptic weight # Synaptic weight from L4e to L2/3e is doubled
std_w_ex = 0.1 * w_ex  # standard deviation weight
g = 4
bg_freq = 8
defaultclock.dt = 0.1 * ms  # timestep of numerical integration method
tsim = 0.1  # Simulation time

# Leaky integrate-and-fire model equations
# dv/dt: equation 1 from the article
# dI/dt: equation 2 from the article
LIFmodel = '''
	dv/dt = (-v + v_r)/tau_m + (I+Iext)/Cm : volt (unless refractory)
	dI/dt = -I/tau_syn : amp
	Iext : amp
	'''
# Reset condition
resetLIF = '''
	v = v_r
	'''

syn_model = '''
				w:amp			# synaptic weight
				'''

# equations executed only when pre-synaptic spike occurs:
# for excitatory connections
pre_eq = '''
			I_post += w
			'''

##          2/3e   2/3i    4e    4i     5e    5i   6e     6i
lname = ['2/3e', '2/3i', '4e', '4i', '5e', '5i', '6e', '6i']
'''n_layer = [10336, 2976, 2432, 544, 10944, 2752, 7200, 1472] # Number of neurons in each group
bg_layer = [2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850] # Neurons for cortical input'''

tha_layer = [5, 5]  # Number of neurons in each THA "unit"
num_tha_groups = len(tha_layer)  # Number of THA units
Ntha = sum(tha_layer)  # Number of THA neurons

n_layer = [103, 29, 24, 5, 109, 27, 72, 14]  # Smaller neuron groups to make simulation faster
bg_layer = [20, 18, 20, 18, 20, 18, 20, 18]

num_groups = len(n_layer)  # Number of different neuron groups (= 8)
N = sum(n_layer)  # The total amount of neurons (sum of each group)
nn_cum = [0]  #
nn_cum.extend(cumsum(n_layer))  # Extends the 'nn_cum' list with 'cumsum' funct of 'n_layer
# cumsum() returns cumulative cum of the elements along given axis - What is the purpose of this?

# Prob. connection table
table = array([[0.192, 0.309, 0.334, 0.585, 0.014, 0, 0.016, 0],
               [0.252, 0.255, 0.255, 0.422, 0.034, 0, 0.008, 0],
               [0.016, 0.012, 0.371, 0.775, 0.003, 0, 0.088, 0],
               [0.133, 0.006, 0.524, 0.833, 0.001, 0, 0.201, 0],
               [0.19, 0.12, 0.377, 0.06, 0.038, 0.166, 0.04, 0],
               [0.107, 0.053, 0.212, 0.02, 0.027, 0.137, 0.018, 0],
               [0.032, 0.014, 0.175, 0.162, 0.026, 0.008, 0.078, 0.399],
               [0.071, 0.002, 0.027, 0.01, 0.012, 0.003, 0.128, 0.267]])  # Probs for connections WITHOUT THA neurons

'''table = array([[0.192, 0.309, 0.334, 0.585, 0.014, 0, 0.016, 0],
               [0.252, 0.255, 0.255, 0.422, 0.034, 0, 0.008, 0],
               [0.016, 0.012, 0.371, 0.775, 0.003, 0, 0.088, 0],
               [0.133, 0.006, 0.524, 0.833, 0.001, 0, 0.201, 0],
               [0.19, 0.12, 0.377, 0.06, 0.038, 0.166, 0.04, 0],
               [0.107, 0.053, 0.212, 0.02, 0.027, 0.137, 0.018, 0],
               [0.032, 0.014, 0.175, 0.162, 0.026, 0.008, 0.078, 0.399],
               [0.071, 0.002, 0.027, 0.01, 0.012, 0.003, 0.128, 0.267]]) '''  # Probs for connections WITH THA neurons?

###############################################################################
###########                   Create Cortical Neurons               ###########
###############################################################################
neurons = NeuronGroup(N, LIFmodel, threshold='v>v_th', reset=resetLIF, \
                      method='linear', refractory=tau_ref)  # Creates a neuron group with N neurons

# Setting initial values for membrane potential and currents
neurons.v = '-58.0*mV + 10.0*mV*randn()'  # "randn" generates data from the "normal distribution" mean=0, sd=1

neurons.I = 0 * pA  # initial value for synaptic currents
neurons.Iext = 0 * pA  # constant external current

pop = []  # Stores NeuronGroups, one for each population
for r in range(0, num_groups):
    pop.append(neurons[nn_cum[r]:nn_cum[r + 1]])

###############################################################################
###########                   Create Thalamus Neurons               ###########
###############################################################################
T = 5  # Number of THA neurons in one unit
# One neuron group defined for each unit, 2 units to start with
# THA unit 1
tha_neurons1 = NeuronGroup(T, LIFmodel, threshold='v>v_th', reset=resetLIF,
                           method='linear', refractory=tau_ref)  # Defined in the same way as MCx neurons

# setting initial values for membrane potential and currents of thalamic and TRN neurons
tha_neurons1.v = '-58.0*mV + 10.0*mV*randn()'  # "randn" generates data from the "normal distribution" mean=0, sd=1

tha_neurons1.I = 10 * pA  # initial value for synaptic current of tha neurons
tha_neurons1.Iext = 10 * pA  # constant external current

# THA unit 2
tha_neurons2 = NeuronGroup(T, LIFmodel, threshold='v>v_th', reset=resetLIF,
                           method='linear', refractory=tau_ref)  # Defined in the same way as MCx neurons

# setting initial values for membrane potential and currents of thalamic and TRN neurons
tha_neurons2.v = '-58.0*mV + 10.0*mV*randn()'  # "randn" generates data from the "normal distribution" mean=0, sd=1

tha_neurons2.I = 10 * pA  # initial value for synaptic current of tha neurons
tha_neurons2.Iext = 10 * pA  # constant external current

###############################################################################
###########                   Create Thalamic Synapses              ###########
###############################################################################
# Recurrent connections between thalamus and cortex, and cortex and thalamus for each neuron unit
# First THA unit
tha1ct = Synapses(tha_neurons1, pop[4], model=syn_model, on_pre=pre_eq)
tha1ct.connect()
tha1tc = Synapses(pop[4], tha_neurons1, model=syn_model, on_pre=pre_eq)
tha1tc.connect()
# Second THA unit
tha2ct = Synapses(tha_neurons2, pop[4], model=syn_model, on_pre=pre_eq)
tha2ct.connect()
tha2tc = Synapses(pop[4], tha_neurons2, model=syn_model, on_pre=pre_eq)
tha2tc.connect()

###############################################################################
###########                   Create Cortical Synapses              ###########
###############################################################################

con = []  # Stores connections
nsyn_array = np.zeros_like(table)  # Stores numbers of synapses per connection

# Connecting neurons
pre_index = []
post_index = []

for c in range(0, num_groups):
    for r in range(0, num_groups):

        nsyn = int(log(1.0 - table[r][c]) / log(
            1.0 - (1.0 / float(n_layer[c] * n_layer[r]))))  # Peter's rule for synapse connectivity
        nsyn_array[r][c] = nsyn

        pre_index = randint(n_layer[c], size=nsyn)
        post_index = randint(n_layer[r], size=nsyn)

        if nsyn < 1:
            pass
        else:
            # Excitatory connections
            if (c % 2) == 0:
                # Synaptic weight from L4e to L2/3e is doubled
                if c == 2 and r == 0:
                    con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                    con[-1].connect(i=pre_index, j=post_index)
                    con[-1].w = '2.0*clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                else:
                    con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                    con[-1].connect(i=pre_index, j=post_index)
                    con[-1].w = 'clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                con[-1].delay = 'clip(d_ex + std_d_ex*randn(), 0.1*ms, d_ex*inf)'

            # Inhibitory connections
            else:
                con.append(Synapses(pop[c], pop[r], model=syn_model, on_pre=pre_eq))
                con[-1].connect(i=pre_index, j=post_index)
                con[-1].w = '-g*clip((w_ex + std_w_ex*randn()),w_ex*0.0, w_ex*inf)'
                con[-1].delay = 'clip(d_in + std_d_in*randn(), 0.1*ms, d_in*inf)'


###############################################################################
###########                   Selection Matrices                    ###########
###############################################################################
def selection_matrix(dim, place):  # Function builds a nxn diagonal matrix with 1 in specific diagonal location
    S = np.matrix(np.zeros((dim, dim), dtype=np.int))
    position_row = place
    position_col = place
    position = [int(dim * position_row + position_col)]
    np.put(S, position, 1)
    return S


S1 = selection_matrix(2, 0)  # Selection matrix for THA unit 1
S2 = selection_matrix(2, 1)  # Selection matrix for THA unit 2

###############################################################################
###########                   Define Input                          ########### 
###############################################################################
bg_in = []
for r in range(0, num_groups):
    bg_in.append(PoissonInput(pop[r], 'I', bg_layer[r], bg_freq * Hz, weight=w_ex))

###############################################################################
###########                      Monitors                           ###########
###############################################################################
# statemon = StateMonitor(neurons, variables = ['v', 'I'], record = [0, 1])

spikemon = SpikeMonitor(neurons)

rmon_net_L23E = PopulationRateMonitor(neurons[:nn_cum[1]])
rmon_net_L23I = PopulationRateMonitor(neurons[nn_cum[1]:nn_cum[2]])
rmon_net_L4E = PopulationRateMonitor(neurons[nn_cum[2]:nn_cum[3]])
rmon_net_L4I = PopulationRateMonitor(neurons[nn_cum[3]:nn_cum[4]])
rmon_net_L5E = PopulationRateMonitor(neurons[nn_cum[4]:nn_cum[5]])
rmon_net_L5I = PopulationRateMonitor(neurons[nn_cum[5]:nn_cum[6]])
rmon_net_L6E = PopulationRateMonitor(neurons[nn_cum[6]:nn_cum[7]])
rmon_net_L6I = PopulationRateMonitor(neurons[nn_cum[7]:nn_cum[8]])

l23e_state = StateMonitor(pop[0], 'v', record=True)  # records state of layer 2/3 excitatory neurons
l4e_state = StateMonitor(pop[2], 'v', record=True)  # records state of layer 4 excitatory neurons
l5e_state = StateMonitor(pop[4], 'v', record=True)  # records state of layer 5 excitatory neurons
l6e_state = StateMonitor(pop[6], 'v', record=True)  # records state of layer 6 excitatory neurons
l5e_spikes = SpikeMonitor(pop[4])  # Spike times of excitatory neurons in l5

# Record from individual neurons
'''import numpy as np
import brian2 as b2
num_neurons = 10  # Not applicable here since neurons are defined above
time_vector = np.arange(0, 30, 5)  # Numpy.ndarray which is the length of the sim time, w/ increments = timestep
frequency_matrix = np.zeros([len(time_vector), num_neurons])  # num rows = (time step*sim time), num cols = num neurons
times = spikemon.t / b2.ms
indices = spikemon.i
timestep = 0.005  # ms - is this same as time step for integration???
for i in range(len(time_vector) - 1):
    time_mask = ((times >= time_vector[i]) & (times <= time_vector[i + 1]))
    indexbin = indices[time_mask]
    for j in range(num_neurons):
        num_count = indexbin.tolist().count(j)
        frequency_matrix[i][j] = num_count / timestep

for k in range(frequency_matrix.shape[1]):
    frequency_matrix[:, k] = scipy.ndimage.gaussian_filter1d(frequency_matrix[:, k], sigma=6)'''
# <Subgroup 'neurongroup_subgroup_4' of 'neurongroup' from 16288 to 27232>
n1 = PopulationRateMonitor(neurons[1])  # random neuron 1
n2 = PopulationRateMonitor(neurons[2])  # Random neuron 2
n3 = PopulationRateMonitor(neurons[3])  # Random neuron 3

###############################################################################
###########                         Run                             ########### 
###############################################################################
net = Network(collect())
net.add(neurons, pop, con, bg_in)  # Adding objects to the simulation
net.run(tsim * second, report='stdout')

###############################################################################
###########                        GPFA                             ###########
###############################################################################
'''def generate_spiketrains(instantaneous_rates, num_trials, timestep):
    """
    Parameters
    ----------
    instantaneous_rates : np.ndarray
        Array containing time series.
    timestep :
        Sample period.
    num_steps : int
        Number of timesteps -> max_time = timestep*(num_steps-1).

    Returns
    -------
    spiketrains : list of neo.SpikeTrains
        List containing spiketrains of inhomogeneous Poisson
        processes based on given instantaneous rates.

    """

    spiketrains = []
    for _ in range(num_trials):
        spiketrains_per_trial = []
        for inst_rate in instantaneous_rates:
            anasig_inst_rate = neo.AnalogSignal(inst_rate, sampling_rate=1 / timestep, units=pq.Hz)
            spiketrains_per_trial.append(inhomogeneous_poisson_process(anasig_inst_rate))
        spiketrains.append(spiketrains_per_trial)

    return spiketrains

bin_size = 10*pq.ms
latent_dimensionality = 2  # The number of dimensions PCA reduces N-dimensional spiking to

gpfa_l5e = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)
l5e_spiketrains = generate_spiketrains(rmon_net_L5E.rate/Hz, 1*ms, 100)
print(l5e_spiketrains)'''

###############################################################################
###########                 Output Measures                         ########### 
###############################################################################

'''plt.figure()    
plt.plot(spikemon.t, spikemon.i, 'k.')
plt.title('Raster Plot')
plt.xlabel('time (s)')
plt.ylabel('neuron index')
#plt.figure()    
#plt.plot(statemon.v[1][1000:1500])

plt.show()'''

# break

'''mean_rates = []
ratemon_list = [rmon_net_L23E, rmon_net_L23I, rmon_net_L4E, rmon_net_L4I, rmon_net_L5E, rmon_net_L5I, rmon_net_L6E, rmon_net_L6I] 

for ratemon in ratemon_list:
    mean_rates.append(np.mean(ratemon.rate[500:]))

print(mean_rates)

L5E_output = rmon_net_L5E.rate[500:]'''

# Plot some 5e neurons
'''x = list(range(10))
for i in x:
    plt.plot(l5e_state.t / ms, l5e_state.v[i])
plt.xlabel('Simulation Time (ms)')
plt.ylabel('Voltage (V)')'''
'''plt.plot(l23e_state.t / ms, l23e_state.v[0], label = 'l2/3e')
plt.plot(l4e_state.t / ms, l4e_state.v[0], label = 'l4e')
plt.plot(l6e_state.t / ms, l6e_state.v[0], label = 'l6e')
plt.plot(l5e_state.t / ms, l5e_state.v[0], label = 'l5e')
plt.plot(rmon_net_L5E.t/ms, rmon_net_L5E.rate/Hz)
plt.legend()
plt.show()'''

print('n1', n1.rate / Hz)

# Plots spike times of neuron i in layer 5e
'''plt.plot(l5e_spikes.t / ms, l5e_spikes.i, '.k', linewidth=0.1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()'''

# Plot low dimensional structure
'''fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(n1.rate / Hz, n2.rate / Hz, n3.rate / Hz, 'gray')
plt.show()'''

# Plot layer 5E activity
plt.plot(rmon_net_L5E.t / ms, rmon_net_L5E.rate / Hz)
plt.xlabel('Time (ms)')
plt.ylabel('L5E Firing rate')
plt.show()

plt.plot(rmon_tha.t / ms, rmon_tha.rate / Hz)
plt.plot(rmon_tha2.t / ms, rmon_tha2.rate / Hz)
plt.show()
