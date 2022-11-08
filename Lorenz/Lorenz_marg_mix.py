import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/Lorenz full")

import data_generate as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

# %% Load data from Margazoglou

data = np.loadtxt('data')

# normalise input data
max_data = data.max(axis=0)
min_data = data.min(axis=0)
norm = max_data - min_data
data = data / norm
training_data, testing_data = data[:44399], data[44399:]

# %% Check and plot training dataset

u, v, w = training_data.transpose()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, s = 5)
plt.show()

# %% Check and plot testing dataset

u, v, w = testing_data.transpose()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, s = 5)
plt.show()

# %% define and train ESN

steps_trans = 444

d = 3
N = 100

ld = 10**(-13) 
gamma = 3.7
spec_rad = 1.2
s = 0
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

np.random.seed(201)

C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.gen_matrix(shape=(N,N), density=0.01, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.spectral_radius_matrix(A, spec_rad)

x_0 = np.zeros(shape=(N,1), dtype=float)

state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d)
reg_result = esn_help.regression_covariance(ld, state_dict, steps_trans)
training = esn_help.training_error(state_dict, reg_result, training_data, steps_trans)
predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, False)

# start of lyap spectrum part
h = 0.005

spectrum_stop_time = 20000
x_data = predicting['states'][:, :spectrum_stop_time]
z_data = predicting['z_predictions'][:, :spectrum_stop_time]
esn_data = [ [x_data[:, t].reshape(N, 1), z_data[:, t].reshape(d, 1)] for t in range(x_data.shape[1]) ]

weights = reg_result[0]

def jacobian_esn_func(t, esn_inputs):
    x_t_1, z_t = esn_inputs
    tanh_term = np.tanh(A @ x_t_1 + gamma * (C @ z_t) + s * zeta)
    outer = 1 - np.power(tanh_term, 2)
    J0 = weights.T  # pf0_px
    J1 = outer * A  # pf_px
    J2 = outer * C * gamma # pf_pz
    return J1 + J2 @ J0

lorenz_esn_spectrum = lyaps_spec.lyapunov_spectrum(esn_data, N, h, "difference", 0, jacobian_esn_func)

# %% check forecasts

plot_u_actual = np.hstack((training['z_actuals'][0], predicting['z_actuals'][0]))
plot_u_predictions = np.hstack((training['z_predictions'][0], predicting['z_predictions'][0]))
                               
plot_v_actual = np.hstack((training['z_actuals'][1], predicting['z_actuals'][1]))
plot_v_predictions = np.hstack((training['z_predictions'][1], predicting['z_predictions'][1]))

plot_w_actual = np.hstack((training['z_actuals'][2], predicting['z_actuals'][2]))
plot_w_predictions = np.hstack((training['z_predictions'][2], predicting['z_predictions'][2]))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
#fig.suptitle("Lorenz predictions against actual")

ax1.plot(plot_u_actual, lw=0.7)
ax1.plot(plot_u_predictions, lw=0.7)
ax1.axvline(len(training_data), color='black', linestyle="--")
ax1.set_ylabel('x')

ax2.plot(plot_v_actual, lw=0.7)
ax2.plot(plot_v_predictions, lw=0.7)
ax2.axvline(len(training_data), color='black', linestyle="--")
ax2.set_ylabel('y')

ax3.plot(plot_w_actual, lw=0.7)
ax3.plot(plot_w_predictions, lw=0.7)
ax3.axvline(len(training_data), color='black', linestyle="--")
ax3.set_ylabel('z')
ax3.set_xlabel('time')

fig.show()

# %% check historical accuracy plots

hist_u_actuals = predicting['z_actuals'][0]
hist_u_predictions = predicting['z_predictions'][0]
esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, 'file_name.pdf')

#%% check historical accuracy plots

hist_v_actuals = predicting['z_actuals'][1]
hist_v_predictions = predicting['z_predictions'][1]
esn_help.hist_accuracy_plot(hist_v_actuals, hist_v_predictions, 'file_name.pdf')

# %% check historical accuracy plots

hist_w_actuals = predicting['z_actuals'][2]
hist_w_predictions = predicting['z_predictions'][2]
esn_help.hist_accuracy_plot(hist_w_actuals, hist_w_predictions, 'file_name.pdf')

# %% historical accuracy plots -- all 3 dimensions
hist_actuals = np.concatenate((predicting['z_actuals'][0], predicting['z_actuals'][1], predicting['z_actuals'][2]), axis=0)
hist_predictions = np.concatenate((predicting['z_predictions'][0], predicting['z_predictions'][1], predicting['z_predictions'][2]), axis=0)
esn_help.hist_accuracy_plot(hist_actuals, hist_predictions, 'file_name.pdf')
