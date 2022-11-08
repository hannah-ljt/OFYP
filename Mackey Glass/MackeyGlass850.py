import numpy as np
import matplotlib.pyplot as plt

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
N = 850

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

# %% check forecasts

z_actuals = np.concatenate((training['z_actuals'], predicting['z_actuals']), axis=1)[0]
z_predictions = np.concatenate((training['z_predictions'], predicting['z_predictions']), axis=1)[0]

t_range = np.arange(step_trans, step_trans + len(z_actuals))

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.axvline(len(training_data), color='black', linestyle="--")
forecast_ax.plot(t_range, z_actuals, label='actual', lw=0.7)
forecast_ax.plot(t_range, z_predictions, label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.legend()

plt.show()

# %% plot lyapunov spectrums

spectrum_stop_time = 8000
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

mg_esn_spectrum = lyaps_spec.lyapunov_spectrum(esn_data, N, 1, "difference", 0, jacobian_esn_func)

# %% generate actual Lyapunov spectrum for lorenz

delay = mg_args['delay']
a = mg_args['a']
b = mg_args['b']
n = mg_args['n']
steps_trans_mg = 1
disc_step = delay
disc = int(delay / h) + 1

def Dx_lagF(z_lag):
    return -a * n * z_lag**n * (1 + z_lag**n)**(-2) + a * (1 + z_lag**n)**(-1)

def jacobian_mg_disc(t, z_k):
    N = int((delay / h)) + 1
    jacobian = np.zeros(shape=(N, N))
    
    for col in range(N-1):
        partial_lag = Dx_lagF(z_k[col])
        power = 0
        for row in range(col, N):
            entry = h * (1 - b*h)**power * partial_lag
            jacobian[row][col] = entry
            power = power + 1
            
    power = 1
    for row_last_col in range(N-1):
        entry = (1 - b*h)**power
        jacobian[row_last_col][N-1] = entry
        power = power + 1
        
    jacobian[N-1][N-1] = (1 - b*h)**N + h * Dx_lagF(z_k[N-1])
    
    return jacobian

mg_spectrum = lyaps_spec.lyapunov_spectrum(mg_data, disc, disc_step, "difference", steps_trans_mg, jacobian_mg_disc)

# %% compare lyapunov spectrums for mackey glass and esn

plt.figure(figsize=(10, 5))

mg_spectrum_sorted = np.sort(mg_spectrum)[::-1]
esn_spectrum_sorted = np.sort(mg_esn_spectrum)[::-1]

mg_idx = np.arange(0, len(mg_spectrum_sorted))
plot_mg = mg_spectrum_sorted
plt.scatter(mg_idx, plot_mg, s=1, marker='o', label='actual')

esn_idx = np.arange(0, len(esn_spectrum_sorted))
plot_mg_esn = esn_spectrum_sorted
plt.scatter(esn_idx, plot_mg_esn, s=1, marker='x', label='ESN')

plt.axhline(c='black', lw=1, linestyle='--')

plt.ylabel('$\lambda$')
plt.xlabel('dimension')
plt.legend()
#plt.savefig('MackeyGlass Spectrums.pdf')
#plt.close()

# %% Kaplan-Yorke Dimension

KY_MG = lyaps_spec.kaplan_yorke(mg_spectrum_sorted)
KY_esn = lyaps_spec.kaplan_yorke(esn_spectrum_sorted)

print(KY_MG, KY_esn)