import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random
from scipy import stats
import seaborn as sns


# Function to generate reservoir

def gen_matrix(shape, density, sd=1, mean=0, loc_seed=100, val_seed=100, pdf="gaussian", seeded=True):
    
    def seeded_rvs_gauss(array_len):
            return stats.norm(loc=mean, scale=sd).rvs(random_state = val_seed, size=array_len)

    def seeded_rvs_uniform(array_len):
        return stats.uniform(loc=mean, scale=sd).rvs(random_state = val_seed, size=array_len)

    m = shape[0]
    n = shape[1]

    if seeded == True:
        
        if pdf == "gaussian":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=seeded_rvs_gauss).A
            return M

        if pdf == "uniform":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=seeded_rvs_uniform).A
            return M

        if pdf == "ones":
            M = random(m, n, density=density, random_state=loc_seed, data_rvs=np.ones).A
            return M
        else: 
            print("No such pdf")
            
    elif seeded == False:
        
        if pdf == "gaussian":
            unseeded_rvs = stats.norm(loc=mean, scale=sd).rvs
            M = random(m, n, density=density, data_rvs=unseeded_rvs).A
            return M

        if pdf == "uniform":
            unseeded_rvs = stats.uniform(loc=mean, scale=sd).rvs
            M = random(m, n, density=density, data_rvs=unseeded_rvs).A
            return M

        if pdf == "ones":
            M = random(m, n, density=density, data_rvs=np.ones).A
            return M
        else: 
            print("No such pdf")
            
    else:
        print("Seeded was neither true nor false")

def spectral_radius(M):
    max_abs_eigenvalue = -1
    eigenvalues, eigenvectors = np.linalg.eig(M)
    for eigenvalue in eigenvalues:
        if abs(eigenvalue) > max_abs_eigenvalue:
            max_abs_eigenvalue = abs(eigenvalue)
    return max_abs_eigenvalue

def spectral_radius_matrix(M, desired_spec_rad):
    M_sr = spectral_radius(M)
    if M_sr == 0:
        return M
    else:
        M = M*(desired_spec_rad/M_sr)
        return M

    
    
# Reservoir equations

def sigma(value):
    return np.tanh(value)

def state(x_prev, z_curr, A, gamma, C, s, zeta, d):
    z_curr = z_curr.reshape(d, 1)
    x_curr = sigma(A @ x_prev + gamma*(C @ z_curr) + s*zeta)
    return x_curr     # outputs (N, 1) array

def observation(x_curr, reg_result):
    w = reg_result[0]
    bias = reg_result[1]
    z_next = np.transpose(w) @ x_curr + bias
    return z_next    # outputs (d, 1) array



# Listening, training and predicting(testing)

def listening(training_data, x_0, A, gamma, C, s, zeta, d):
    state_dict = {'all_states': None,
                  'last_state': None, 
                  'input_data': None}
    
    T = len(training_data)
    
    for t in range(0, T):
        if t == 0:
            x_curr = x_0
            X = np.array(x_curr)
            z_curr = training_data[t].reshape(d, )
            Z = np.array(z_curr)
            
        else:
            x_curr = state(x_curr, z_curr, A, gamma, C, s, zeta, d)
            X = np.column_stack((X, x_curr))
            z_curr = training_data[t].reshape(d, )
            Z = np.column_stack((Z, z_curr))
                
    state_dict['last_state'] = x_curr
    state_dict['all_states'] = X
    state_dict['input_data'] = Z
    
    return state_dict


def regression_sol_alt(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]
   
    N = X.shape[0]
    d = Z.shape[0]
    
    w_best = np.linalg.solve(X @ X.transpose() + ld * np.identity(N), X @ Z.transpose())
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1).reshape(N, 1))).reshape(d, 1)
    
    return w_best, a_best    # outputs (N, d) array and (1, ) array


def regression_sol(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]

    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]
    
    X_concat = np.concatenate((X, np.ones(shape=(1, T))), axis=0)
    X_tranpose_concat = np.concatenate((X.transpose(), np.zeros(shape=(T, 1))), axis=1)
    
    regularisation = ld * np.identity(N)
    zeros_row = np.zeros(shape=(1, N))
    zeros_col = np.zeros(shape=(N+1, 1))
    regularisation_concat = np.concatenate((regularisation, zeros_row), axis=0)
    regularisation_concat = np.concatenate((regularisation_concat, zeros_col), axis=1) 
    regularisation_concat[N][N] = T
    
    reg_best = np.linalg.solve(X_concat @ X_tranpose_concat + regularisation_concat, X_concat @ Z.transpose())
    
    w_best = reg_best[0:N, :]
    a_best = reg_best[N, :].reshape(d, 1)
    
    return w_best, a_best    # outputs same forecast results as regression_sol

def regression_covariance(ld, state_dict, T_trans):
    X = state_dict['all_states'][:, T_trans:]
    Z = state_dict['input_data'][:, T_trans:]
    
    N = X.shape[0]
    T = X.shape[1]
    d = Z.shape[0]
    
    cov_XZ = (1/T) * (X @ Z.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(Z, axis=1).reshape(d, 1).transpose())
    cov_XX = (1/T) * (X @ X.transpose()) - (np.mean(X, axis=1).reshape(N, 1) @ np.mean(X, axis=1).reshape(N, 1).transpose()) + ld * np.identity(N)
    
    w_best = np.linalg.solve(cov_XX, cov_XZ)
    a_best = (np.mean(Z, axis=1) - (w_best.transpose() @ np.mean(X, axis=1))).reshape(d, 1)

    return w_best, a_best
    

def prediction(state_dict, reg_result, testing_data, A, gamma, C, s, zeta, d, platt_err):
    prediction_dict = {'testing_error': None,
                       'z_actuals': None,
                       'z_predictions': None,
                       'states': None}
    
    T_bar = len(testing_data)
    T = state_dict.get('input_data').shape[1]
    
    x_prev = state_dict.get('last_state')
    z_predict = state_dict.get('input_data')[:, -1].reshape(d, 1)
    
    testing_error = 0
    weight_sum = 0

    for t_bar in range(0, T_bar):
        x_prev = state(x_prev, z_predict, A, gamma, C, s, zeta, d)
        z_predict = observation(x_prev, reg_result).reshape(d, 1)
        z_actual = np.array(testing_data[t_bar]).reshape(d, 1)   # is numpy scalar, reshaped to (d, 1)
        
        if t_bar == 0:
            z_predictions = np.array(z_predict)
            z_actuals = np.array(z_actual)
            states = np.array(x_prev)
            
        else:
            z_predictions = np.concatenate((z_predictions, z_predict), axis=1)
            z_actuals = np.concatenate((z_actuals, z_actual), axis=1)
            states = np.concatenate((states, x_prev), axis=1)
            
        if platt_err == True:
            testing_error = testing_error + np.linalg.norm(z_predict - z_actual)*np.exp(-((t_bar-T)/T_bar))
            weight_sum = weight_sum + np.exp(-((t_bar-T)/T_bar))
        else:    
            testing_error = testing_error + np.linalg.norm(z_predict - z_actual)**2
            weight_sum = weight_sum + 1
 
    prediction_dict['testing_error'] = testing_error/weight_sum
    prediction_dict['z_actuals'] = z_actuals
    prediction_dict['z_predictions'] = z_predictions   
    prediction_dict['states'] = states
        
    return prediction_dict


def training_error(state_dict, reg_result, training_data, T_trans):
    training_error_dict = {'training_error': None,
                           'z_actuals': None,
                           'z_predictions': None}

    X = state_dict.get('all_states')
    X = X[:, T_trans:]
    
    Z = state_dict.get('input_data')
    Z = Z[:, T_trans:]
    
    N = X.shape[0]
    d = Z.shape[0]
    
    training_error = 0
    T = len(training_data)
    
    for t in range(0, T-T_trans):
        x_prev = X[:, t].reshape(N, 1)    
        z_predict = observation(x_prev, reg_result).reshape(d, 1)
        z_actual = Z[:, t].reshape(d, 1)   
        
        if t == 0:
            z_predictions = np.array(z_predict)
            z_actuals = np.array(z_actual)
            
        else:
            z_predictions = np.concatenate((z_predictions, z_predict), axis=1)
            z_actuals = np.concatenate((z_actuals, z_actual), axis=1)
        
        training_error = training_error + np.linalg.norm(z_predict - z_actual)**2
        
    training_error = training_error/(T-T_trans)
    
    training_error_dict['training_error'] = training_error
    training_error_dict['z_actuals'] = z_actuals
    training_error_dict['z_predictions'] = z_predictions
        
    return training_error_dict


# Plotting functions

def state_plot(state_dict, plotwith_init, T_trans, node=0):
    X = state_dict.get('all_states')
    if plotwith_init == True:
        state_plot, state_ax = plt.subplots(figsize=(20,5))
        state_ax.plot(X[node][:])
        state_ax.set_title('Plot of States at node {}'.format(node))
        state_ax.set_xlabel('time')
        state_ax.set_ylabel('state of node {}'.format(node))
        state_plot.savefig('state_plot.pdf')
        
        return (np.amin(X[node][:]), np.amax(X[node][:]))
    
    if plotwith_init == False:
        state_plot, state_ax = plt.subplots(figsize=(20,5))
        state_ax.plot(X[node][T_trans:])
        state_ax.set_title('Plot of States at node {}'.format(node))
        state_ax.set_xlabel('time')
        state_ax.set_ylabel('state of node {}'.format(node))
        state_plot.savefig('state_plot.pdf')
    
        return (np.amin(X[node][T_trans:]), np.amax(X[node][T_trans:]))
    
                
def hist_accuracy_plot(actuals, predictions, fname, with_bars=False):
    if with_bars == False:
        sns.kdeplot(actuals, label='actual', shade=True, color='red')
        sns.kdeplot(predictions, label='prediction', shade=True, color='skyblue')
        plt.legend()
        plt.savefig(fname)
    
    if with_bars == True:
        sns.histplot(actuals, label='actual', color='red', kde=True)
        sns.histplot(predictions, label='prediction', color='skyblue', kde=True)
        plt.legend()
        plt.savefig(fname)
        
