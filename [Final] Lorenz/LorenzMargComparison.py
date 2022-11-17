import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from scipy.io import loadmat

import sys
sys.path.append("Users/Henrik/Documents/Hannah/[Final] Lorenz")

import data_generate as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

# %% loading Marg dataset -- 40000, 0.005, prepared my way

lorenz_data = np.loadtxt("/Users/Henrik/Documents/Hannah/Lorenz full/data")
steps_split = 44400
steps_trans = 444
h = 0.005
training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% loading Marg dataset -- 5000, 0.005, prepared my way

lorenz_data = np.loadtxt("/Users/Henrik/Documents/Hannah/Lorenz full/data")
steps_split = 5000
steps_trans = 444
h = 0.005
training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% creating my dataset -- 40000, 0.005 

def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 200)
t_split = 100
t_trans = 2

time_steps, lorenz_data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)

steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% creating my dataset -- 5000, 0.005

def lorenz(t, Z, args):
    u, v, w = Z
    sig, beta, rho = args
    
    up = -sig*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    
    return np.array([up, vp, wp])

lor_args = (10, 8/3, 28)
Z0 = (0, 1, 1.05)

h = 0.005
t_span = (0, 50)
t_split = 25
t_trans = 2

time_steps, lorenz_data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)

steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% creating my dataset -- 40000, 0.02

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
t_span = (0, 800)
t_split = 400
t_trans = 8

time_steps, lorenz_data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)

steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% creating my dataset -- 5000, 0.02

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
t_span = (0, 100)
t_split = 50
t_trans = 8

time_steps, lorenz_data = data_gen.rk45(lorenz, t_span, Z0, h, lor_args)

steps_trans = int(t_trans/h)
steps_split = int(t_split/h)

training_data, testing_data = 0.01*lorenz_data[:steps_split], 0.01*lorenz_data[steps_split:]

# %% plot and check datasets

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
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# %% check testing data
u, v, w = testing_data.transpose()

start = 0
stop = len(testing_data) #int(50/h)
step = 1

plot_ls = time_steps[start:stop:step]

u_plot = u[start:stop:step]
v_plot = v[start:stop:step]
w_plot = w[start:stop:step]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_plot, v_plot, w_plot, c=plot_ls, cmap='viridis', s = 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# %% My ESN

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

%time state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d, N)
%time reg_result = esn_help.regression_covariance(ld, state_dict, steps_trans)
%time predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, N)

training_error = esn_help.training_error(state_dict, reg_result, steps_trans)
print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

# %% their ESN

d = 3
N = 100

# load their dictionary containing the ESN information
parameters = loadmat('/Users/Henrik/Documents/Hannah/[Final] Lorenz/ESN_RVC_Noise_dt_0.005_noise_0.0005_ens_2')
Win = parameters['Win']
W = parameters['W']
Wout = parameters['Wout']
norm = parameters['norm'][0]
fix_hyp = parameters['fix_hyp'][0]
opt_hyp = parameters['opt_hyp'][0]

# convert the information to ESN for our functions
ld = opt_hyp[2].copy()
gamma = opt_hyp[1].copy()
rho = opt_hyp[0].copy()
A = W[0][1].toarray()
C = Win[0][1].toarray()[:, 0:d]
zeta = Win[0][1].toarray()[:, d].reshape(-1, 1)
s = 1

x_0 = np.zeros(shape=(N,1), dtype=float)

%time state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d, N)
%time reg_result = esn_help.regression_covariance(ld, state_dict, steps_trans)
%time predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, N)

training_error = esn_help.training_error(state_dict, reg_result, steps_trans)
print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

# %% # %% check forecasts

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), tight_layout=True)
#fig.suptitle("Lorenz predictions against actual")

ax1.plot(predicting['z_actuals'][0], lw=0.7)
ax1.plot(predicting['z_predictions'][0], lw=0.7)
ax1.set_ylabel('x')

ax2.plot(predicting['z_actuals'][1], lw=0.7)
ax2.plot(predicting['z_predictions'][1], lw=0.7)
ax2.set_ylabel('y')

ax3.plot(predicting['z_actuals'][2], lw=0.7)
ax3.plot(predicting['z_predictions'][2], lw=0.7)
ax3.set_ylabel('z')
ax3.set_xlabel('time')

fig.show()

# %% check distribution accuracy plots

hist_u_actuals = predicting['z_actuals'][0]
hist_u_predictions = predicting['z_predictions'][0]
esn_help.hist_accuracy_plot(hist_u_actuals, hist_u_predictions, x_label='x', y_label='frequency')

#%% check distribution accuracy plots

hist_v_actuals = predicting['z_actuals'][1]
hist_v_predictions = predicting['z_predictions'][1]
esn_help.hist_accuracy_plot(hist_v_actuals, hist_v_predictions, x_label='y', y_label='frequency')

# %% check distribution accuracy plots

hist_w_actuals = predicting['z_actuals'][2]
hist_w_predictions = predicting['z_predictions'][2]
esn_help.hist_accuracy_plot(hist_w_actuals, hist_w_predictions, x_label='z', y_label='frequency')

# %% check distribution accuracy plots -- all 3 dimensions

hist_actuals = np.concatenate((predicting['z_actuals'][0], predicting['z_actuals'][1], predicting['z_actuals'][2]), axis=0)
hist_predictions = np.concatenate((predicting['z_predictions'][0], predicting['z_predictions'][1], predicting['z_predictions'][2]), axis=0)
esn_help.hist_accuracy_plot(hist_actuals, hist_predictions, x_label='all', y_label='frequency')

# %% plot lyapunov spectrums

x_data = predicting['states'].transpose()[0:5000, :]
weights = reg_result[0]
N_transient = 500

def jacobian_esn_func(t, x_t_1):
    outer = (1 - np.power(x_t_1, 2)).reshape(N, 1)
    J0 = weights.T  # pf0_px
    J1 = outer * A  # pf_px
    J2 = outer * C * gamma # pf_pz
    return J1 + J2 @ J0

%time lorenz_esn_spectrum = lyaps_spec.lyapunov_spectrum(x_data, N, h, "difference", N_transient, jacobian_esn_func)

print(lorenz_esn_spectrum)

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