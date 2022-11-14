import numpy as np
import matplotlib.pyplot as plt
import nolds

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/Mackey Glass full")

import data_generate_delay as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

# %% Prepare dataset

def mackeyglass(t, z, z_lag, mg_args):
    
    a = mg_args['a']
    b = mg_args['b']
    n = mg_args['n']
    
    return (a * z_lag) / (1 + z_lag**n) - b*z

def init(t):
    return 1.2

mg_args = {'delay': 17,
           'a': 0.2, 
           'b': 0.1, 
           'n': 10 }

h = 0.02
n_intervals = 700

time_steps, mg_data = data_gen.dde_rk45(n_intervals, init, mackeyglass, h, mg_args)

slicing = int(1 / h)
step_split = 3000
step_trans = 1000

time_sliced, mg_sliced = time_steps.flatten()[::slicing], mg_data.flatten()[::slicing]

mg_sliced = np.tanh(mg_sliced - 1)
training_data, testing_data = mg_sliced[:step_split], mg_sliced[step_split:]

# %% Check and plot dataset

mg_plot, mg_ax = plt.subplots(figsize=(100, 10))
mg_ax.axvline(len(training_data), color='black', linestyle="--")
mg_ax.plot(time_sliced, mg_sliced, lw=0.7)
mg_ax.set_xlabel('time')
plt.show()

# %% define and train ESN

d = 1
N = 2000

ld = 10**(-13) 
gamma = 1
spec_rad = 0.95
s = 0
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

np.random.seed(304)

C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.gen_matrix(shape=(N,N), density=0.005, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.spectral_radius_matrix(A, spec_rad)

x_0 = np.zeros(shape=(N,1), dtype=float)

state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d)
reg_result = esn_help.regression_covariance(ld, state_dict, step_trans)
training = esn_help.training_error(state_dict, reg_result, training_data, step_trans)
predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, False)

# %% Lyapunov spectrum using nolds

data = predicting['z_predictions'].flatten(order="C")
esn_spectrum = nolds.lyap_e(data, emb_dim=3999, matrix_dim=2000)
data_spectrum = nolds.lyap_e(training_data, emb_dim=1699, matrix_dim=850)

# %% plot spectrum

esn_spectrum_sorted = sorted(esn_spectrum, reverse=True)
data_spectrum_sorted = sorted(data_spectrum, reverse=True)

plt.figure(figsize=(20, 20))

actual_plot_range = data_spectrum_sorted
actual_index = np.arange(0, len(actual_plot_range))
plt.scatter(actual_index, actual_plot_range, s=2, marker='o', label='actual')

esn_plot_range = esn_spectrum_sorted
esn_index = np.arange(0, len(esn_plot_range))
plt.scatter(esn_index, esn_plot_range, s=2, marker='x', label='ESN')

plt.axhline(c='black', lw=1, linestyle='--')

plt.ylabel('$\lambda$')
plt.xlabel('dimension')
plt.title('Comparison of actual and ESN Lyapunov exponents $\lambda$')
plt.legend()

plt.show()
                             
        