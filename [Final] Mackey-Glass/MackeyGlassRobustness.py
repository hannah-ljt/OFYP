import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/[Final] Mackey-Glass")

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
training_data, testing_data = mg_sliced[:step_split], mg_sliced[step_split:5000]

# %% Check and plot dataset

mg_plot, mg_ax = plt.subplots(figsize=(100, 10))
mg_ax.plot(time_sliced, mg_sliced, lw=0.7)
mg_ax.set_ylabel('z')
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

%time state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d, N)
%time reg_result = esn_help.regression_covariance(ld, state_dict, step_trans)
%time predicting = esn_help.prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, N)

training_error = esn_help.training_error(state_dict, reg_result, step_trans)
print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

# %% check forecasts

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.plot(predicting['z_actuals'][0], label='actual', lw=0.7)
forecast_ax.plot(predicting['z_predictions'][0], label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.set_ylabel('z')
forecast_ax.legend()

plt.show()

# %% Save optimised weight
np.savetxt('prev_optimised_weight', reg_result[0])
np.savetxt('prev_optimised_bias', reg_result[1])

# %% constant init func 1.3

def init_2(t):
    return 1.3

time_steps_2, mg_data_2 = data_gen.dde_rk45(n_intervals, init_2, mackeyglass, h, mg_args)
time_sliced_2, mg_sliced_2 = time_steps_2.flatten()[::slicing], mg_data_2.flatten()[::slicing]
mg_sliced_2 = np.tanh(mg_sliced_2 - 1)
training_data_2, testing_data_2 = mg_sliced_2[:step_split], mg_sliced_2[step_split:5000]

state_dict_2 = esn_help.listening(training_data_2, x_0, A, gamma, C, s, zeta, d, N)
reg_result_2 = (np.loadtxt('prev_optimised_weight').reshape(N,1), np.loadtxt('prev_optimised_bias').reshape(1, ))
predicting_2 = esn_help.prediction(state_dict_2, reg_result_2, testing_data_2, A, gamma, C, s, zeta, d, N)
training_2 = esn_help.training_error(state_dict_2, reg_result_2, step_trans)

print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.plot(predicting_2['z_actuals'][0], label='actual', lw=0.7)
forecast_ax.plot(predicting_2['z_predictions'][0], label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.set_ylabel('z')
forecast_ax.legend()

plt.show()

# %% constant init func 1.6

def init_3(t):
    return 1.6

time_steps_3, mg_data_3 = data_gen.dde_rk45(n_intervals, init_3, mackeyglass, h, mg_args)
time_sliced_3, mg_sliced_3 = time_steps_3.flatten()[::slicing], mg_data_3.flatten()[::slicing]
mg_sliced_3 = np.tanh(mg_sliced_3 - 1)
training_data_3, testing_data_3 = mg_sliced_3[:step_split], mg_sliced_3[step_split:5000]

state_dict_3 = esn_help.listening(training_data_3, x_0, A, gamma, C, s, zeta, d, N)
reg_result_3 = (np.loadtxt('prev_optimised_weight').reshape(N,1), np.loadtxt('prev_optimised_bias').reshape(1, ))
training_3 = esn_help.training_error(state_dict_3, reg_result_3, step_trans)
predicting_3 = esn_help.prediction(state_dict_3, reg_result_3, testing_data_3, A, gamma, C, s, zeta, d, N)

print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.plot(predicting_3['z_actuals'][0], label='actual', lw=0.7)
forecast_ax.plot(predicting_3['z_predictions'][0], label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.set_ylabel('z')
forecast_ax.legend()

plt.show()

# %% piecewise constant init func

def init_4(t):
    if 0 <= t % 100 < 20:
        return 1.5
    elif 20 <= t % 100 < 50:
        return 1
    elif 50 <= t % 100 < 100:
        return 1.2
    
time_steps_4, mg_data_4 = data_gen.dde_rk45(n_intervals, init_4, mackeyglass, h, mg_args)
time_sliced_4, mg_sliced_4 = time_steps_4.flatten()[::slicing], mg_data_4.flatten()[::slicing]
mg_sliced_4 = np.tanh(mg_sliced_4 - 1)
training_data_4, testing_data_4 = mg_sliced_4[:step_split], mg_sliced_4[step_split:5000]

state_dict_4 = esn_help.listening(training_data_4, x_0, A, gamma, C, s, zeta, d, N)
reg_result_4 = (np.loadtxt('prev_optimised_weight').reshape(N,1), np.loadtxt('prev_optimised_bias').reshape(1, ))
training_4 = esn_help.training_error(state_dict_4, reg_result_4, step_trans)
predicting_4 = esn_help.prediction(state_dict_4, reg_result_4, testing_data_4, A, gamma, C, s, zeta, d, N)

print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.plot(predicting_4['z_actuals'][0], label='actual', lw=0.7)
forecast_ax.plot(predicting_4['z_predictions'][0], label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.set_ylabel('z')
forecast_ax.legend()

plt.show()

# %% non-constant init func

def init_5(t):
    if 0 <= t % 100 < 30:
        return 0.001*(t%100)
    elif 30 <= t % 100 < 60:
        return -0.001*(t%100)+0.04
    elif 60 <= t % 100 < 100:
        return 0.001*(t%100)-0.06
    
time_steps_5, mg_data_5 = data_gen.dde_rk45(n_intervals, init_5, mackeyglass, h, mg_args)
time_sliced_5, mg_sliced_5 = time_steps_5.flatten()[::slicing], mg_data_5.flatten()[::slicing]
mg_sliced_5 = np.tanh(mg_sliced_5 - 1)
training_data_5, testing_data_5 = mg_sliced_5[:step_split], mg_sliced_5[step_split:5000]

state_dict_5 = esn_help.listening(training_data_5, x_0, A, gamma, C, s, zeta, d, N)
reg_result_5 = (np.loadtxt('prev_optimised_weight').reshape(N,1), np.loadtxt('prev_optimised_bias').reshape(1, ))
training_5 = esn_help.training_error(state_dict_5, reg_result_5, step_trans)
predicting_5 = esn_help.prediction(state_dict_5, reg_result_5, testing_data_5, A, gamma, C, s, zeta, d, N)

print('Training error: ', training_error)
print('Testing error: ', predicting['testing_error'])

forecast_plot, forecast_ax = plt.subplots(figsize=(50, 10))
forecast_ax.plot(predicting_5['z_actuals'][0], label='actual', lw=0.7)
forecast_ax.plot(predicting_5['z_predictions'][0], label='prediction', lw=0.7)
forecast_ax.set_xlabel('time')
forecast_ax.set_ylabel('z')
forecast_ax.legend()

plt.show()

# %% histogram 1.2

z_actuals = predicting['z_actuals'][0]
z_predictions = predicting['z_predictions'][0]
esn_help.hist_accuracy_plot(z_actuals, z_predictions, 'z', 'frequency')

# %% histogram 1.3

z_actuals_2_hist = predicting_2['z_actuals'][0]
z_predictions_2_hist = predicting_2['z_predictions'][0]
esn_help.hist_accuracy_plot(z_actuals_2_hist, z_predictions_2_hist, 'z', 'frequency')

# %% histogram 1.6

z_actuals_3_hist = predicting_3['z_actuals'][0]
z_predictions_3_hist = predicting_3['z_predictions'][0]
esn_help.hist_accuracy_plot(z_actuals_3_hist, z_predictions_3_hist, 'z', 'frequency')

# %% histogram piecewise constant

z_actuals_4_hist = predicting_4['z_actuals'][0]
z_predictions_4_hist = predicting_4['z_predictions'][0]
esn_help.hist_accuracy_plot(z_actuals_4_hist, z_predictions_4_hist, 'z', 'frequency')

# %% histogram not piecewise constant

z_actuals_5_hist = predicting_5['z_actuals'][0]
z_predictions_5_hist = predicting_5['z_predictions'][0]
esn_help.hist_accuracy_plot(z_actuals_5_hist, z_predictions_5_hist, 'z', 'frequency')

# %% plot init_4

vals_4 = [ init_4(t) for t in range(0, 900) ]
plt.plot(vals_4)
plt.xlabel('time')
plt.ylabel('z')

# %% plot init_5

vals_5 = [ init_5(t) for t in range(0, 900) ]
plt.plot(vals_5)
plt.xlabel('time')
plt.ylabel('z')