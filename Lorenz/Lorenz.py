import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/Lorenz full")

import data_generate as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

# %% Prepare dataset
def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.02
t_span = (0, 500)
t_split = 100
t_trans = 20

time_steps, lorenz_data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)

steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% Check and plot training dataset

# check training data
u, v, w = training_data.transpose()

start = steps_trans
stop = len(training_data) #int(50/h)
step = 1

plot_ls = time_steps[start:stop:step]

u_plot = u[start:stop:step]
v_plot = v[start:stop:step]
w_plot = w[start:stop:step]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
plt.show()

# %% check testing data
u, v, w = testing_data.transpose()

start = len(training_data) 
stop = len(testing_data) #int(50/h)
step = 1

plot_ls = time_steps[start:stop:step]

u_plot = u[start:stop:step]
v_plot = v[start:stop:step]
w_plot = w[start:stop:step]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
plt.show()

# %% check time series
u, v, w = lorenz_data.transpose()

start = 0
stop = len(lorenz_data)
step = 1

u_plot = u[start:stop:step]
v_plot = v[start:stop:step]
w_plot = w[start:stop:step]

plot_ls = time_steps[start:stop:step]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
ax1.axvline(t_split, color='black', linestyle="--")
ax1.scatter(time_steps[start:stop:step], u_plot, c=plot_ls, cmap='viridis', s=2)
ax1.set_ylabel('u')
ax2.scatter(time_steps[start:stop:step], v_plot, c=plot_ls, cmap='viridis', s=2)
ax2.axvline(t_split, color='black', linestyle="--")
ax2.set_ylabel('v')
ax3.scatter(time_steps[start:stop:step], w_plot, c=plot_ls, cmap='viridis', s=2)
ax3.axvline(t_split, color='black', linestyle="--")
ax3.set_ylabel('w')
ax3.set_xlabel('time')

plt.show()

# %% define and train ESN

d = 3
N = 2000

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

%time state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d, N)
%time reg_result = esn_help.regression_covariance(ld, state_dict, steps_trans)
%time predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, N)

training_error = esn_help.training_error(state_dict, reg_result, steps_trans)
print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

# %% check forecasts
cut_times = time_steps[len(training_data):]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
#fig.suptitle("Lorenz predictions against actual")

ax1.plot(cut_times, predicting['z_actuals'][0], lw=0.7)
ax1.plot(cut_times, predicting['z_predictions'][0], lw=0.7)
ax1.set_ylabel('x')

ax2.plot(cut_times, predicting['z_actuals'][1], lw=0.7)
ax2.plot(cut_times, predicting['z_predictions'][1], lw=0.7)
ax2.set_ylabel('y')

ax3.plot(cut_times, predicting['z_actuals'][2], lw=0.7)
ax3.plot(cut_times, predicting['z_predictions'][2], lw=0.7)
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

# %% plot lyapunov spectrums

x_data = predicting['states'].transpose()
weights = reg_result[0]
N_transient = 1000

def jacobian_esn_func(t, x_t_1):
    outer = (1 - np.power(x_t_1, 2)).reshape(N, 1)
    J0 = weights.T  # pf0_px
    J1 = outer * A  # pf_px
    J2 = outer * C * gamma # pf_pz
    return J1 + J2 @ J0

%time lorenz_esn_spectrum = lyaps_spec.lyapunov_spectrum(x_data, N, h, "difference", N_transient, jacobian_esn_func)

# %% generate actual Lyapunov spectrum for lorenz

steps_trans_lor = int(20/h)

def jacobian_lorenz(t, lor_inputs):
    u, v, w = lor_inputs
    sig, beta, rho = 10, 8/3, 28
    
    dF1dinputs = [-sig, sig, 0]
    dF2dinputs = [rho-w, -1, -u]
    dF3dinputs = [v, u, -beta]
    
    return np.array([dF1dinputs, dF2dinputs, dF3dinputs])

%time lorenz_spectrum = lyaps_spec.lyapunov_spectrum(lorenz_data, d, h, "ordinary", steps_trans_lor, jacobian_lorenz)

# %% plot spectrums

plt.figure(figsize=(10, 5))

lor_spectrum_sorted = np.sort(lorenz_spectrum)[::-1]
esn_spectrum_sorted = np.sort(lorenz_esn_spectrum)[::-1]

mg_idx = np.arange(0, len(lor_spectrum_sorted))
plot_mg = lor_spectrum_sorted
plt.scatter(mg_idx, plot_mg, s=10, marker='o', label='actual')

esn_idx = np.arange(0, len(esn_spectrum_sorted))
plot_mg_esn = esn_spectrum_sorted
plt.scatter(esn_idx, plot_mg_esn, s=0.7, marker='x', label='ESN')

plt.axhline(c='black', lw=1, linestyle='--')

plt.ylabel('$\lambda$')
plt.xlabel('dimension')
plt.legend()

# %% Kaplan-Yorke dimension

KY_Lor = lyaps_spec.kaplan_yorke(lor_spectrum_sorted)
KY_esn = lyaps_spec.kaplan_yorke(esn_spectrum_sorted)

print(KY_Lor, KY_esn)
