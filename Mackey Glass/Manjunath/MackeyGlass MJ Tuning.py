import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/Mackey Glass Manjunath")

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
#training_data, testing_data = mg_sliced[:step_split], mg_sliced[step_split:]
training_data, testing_data = mg_sliced[:step_split], mg_sliced[step_split:5000]
# %% Check and plot dataset

mg_plot, mg_ax = plt.subplots(figsize=(100, 10))
mg_ax.axvline(len(training_data), color='black', linestyle="--")
mg_ax.plot(time_sliced, mg_sliced, lw=0.7)
mg_ax.set_xlabel('time')
plt.show()

# %% train ESN using manjunath concatenated states

d = 1
N = 2000

ld = 10**(-11) #10**(-6) 
gamma = 0.6 #1.5
spec_rad = 1.1
s = 0
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

np.random.seed(304)

C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.gen_matrix(shape=(N,N), density=0.005, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.spectral_radius_matrix(A, spec_rad)

x_0 = np.zeros(shape=(N,1), dtype=float)

state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d)
reg_result = esn_help.regression_mj(ld, state_dict, step_trans)
training = esn_help.training_error_mj(state_dict, reg_result, training_data, step_trans)
predicting = esn_help.prediction_mj(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, False)

# %% tuning esn 

d = 1
N = 1000

s = 0
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

x_0 = np.zeros(shape=(N,1), dtype=float)

# %% total range

ld_range = np.arange(1, 20, 1)
gamma_range = np.arange(0.1, 5, 0.1)
spec_rad_range = np.arange(0.1, 10, 0.1)

# %% MG_mj1
ld_range = np.arange(1, 6, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj2
ld_range = np.arange(6, 11, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj3
ld_range = np.arange(11, 16, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj4
ld_range = np.arange(16, 21, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj5
ld_range = np.arange(1, 6, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj6
ld_range = np.arange(6, 11, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj7
ld_range = np.arange(11, 16, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj8
ld_range = np.arange(16, 21, 1)
gamma_range = np.arange(0.1, 2.6, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj9
ld_range = np.arange(1, 6, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj10
ld_range = np.arange(6, 11, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj11
ld_range = np.arange(11, 16, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj12
ld_range = np.arange(16, 21, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(0.1, 5.1, 0.1)

# %% MG_mj13
ld_range = np.arange(1, 6, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj14
ld_range = np.arange(6, 11, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj15
ld_range = np.arange(11, 16, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %% MG_mj16
ld_range = np.arange(16, 21, 1)
gamma_range = np.arange(2.6, 5.1, 0.1)
spec_rad_range = np.arange(5.1, 10.1, 0.1)

# %%
print("total # results ", len(ld_range)*len(gamma_range)*len(spec_rad_range))
tuning_results = {"lambda": [], 
                  "gamma": [],
                  "spec_rad":[], 
                  "testing_error":[]}
trial = 0
np.random.seed(304)

C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
A = esn_help.gen_matrix(shape=(N,N), density=0.005, sd=2, mean=-1, pdf="uniform", seeded=False)

for ld_val in ld_range:
    for gamma_val in gamma_range:
        for spec_rad_val in spec_rad_range:
            
            ld = 10**(-1*float(ld_val))
            gamma = gamma_val
            spec_rad = spec_rad_val
            
            A = esn_help.spectral_radius_matrix(A, spec_rad)
            
            state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d)
            reg_result = esn_help.regression_covariance_mj(ld, state_dict, step_trans)
            
            predicting = esn_help.prediction_mj(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, False)
            
            tuning_results['lambda'].append(ld)
            tuning_results['gamma'].append(gamma)
            tuning_results['spec_rad'].append(spec_rad)
            tuning_results['testing_error'].append(predicting['testing_error'])
            
            trial = trial + 1
            if trial % 10 == 0:
                print(trial, predicting['testing_error'])
                
# %% 
                
d = 1
N = 2000

ld = 10**(-6) 
gamma = 1.5
spec_rad = 1.1
s = 0
zeta = esn_help.gen_matrix(shape=(N,1), density=1, pdf="ones", seeded=False)

for seed_val in range(300, 500):
    
    np.random.seed(seed_val)
    
    C = esn_help.gen_matrix(shape=(N,d), density=1, sd=2, mean=-1, pdf="uniform", seeded=False)
    A = esn_help.gen_matrix(shape=(N,N), density=0.005, sd=2, mean=-1, pdf="uniform", seeded=False)
    A = esn_help.spectral_radius_matrix(A, spec_rad)
    
    x_0 = np.zeros(shape=(N,1), dtype=float)
    
    state_dict = esn_help.listening(training_data, x_0, A, gamma, C, s, zeta, d)
    reg_result = esn_help.regression_mj(ld, state_dict, step_trans)
    training = esn_help.training_error_mj(state_dict, reg_result, training_data, step_trans)
    predicting = esn_help.prediction_mj(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, False)
    print(predicting['testing_error'])
    
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
