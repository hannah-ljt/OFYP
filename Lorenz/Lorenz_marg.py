import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import sys
sys.path.append("/Users/Henrik/Documents/Hannah/Lorenz full")

import data_generate as data_gen
import ESN_helperfunctions as esn_help
import lyapunov_spectrum as lyaps_spec

from scipy.io import loadmat

# %% Load data from Margazoglou

U = np.loadtxt('data')

# %% Check and plot training dataset

u, v, w = U.transpose()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u, v, w, s = 5)
plt.show()

# %% define ESN equations to the way Margozoglou did it
def step(x, z, sigma_in, rho):
    z_augmented = np.hstack((z/norm, bias_in))
    x_post = np.tanh(C.dot(z_augmented*sigma_in) + A.dot(rho*x))
    x_augmented = np.concatenate((x_post, bias_out))
    return x_augmented

def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N = U.shape[0]
    N_units = x0.shape[0]
    Xa = np.empty((N+1, N_units+1))
    Xa[0] = np.concatenate((x0, bias_out))
    for i in np.arange(1,N+1):
        Xa[i] = step(Xa[i-1,:N_units], U[i-1], sigma_in, rho)
        
    return Xa

def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    
    xa = x0.copy()
    Yh = np.empty((N+1, dim))
    Yh[0] = np.dot(xa, Wout)    
    
    for i in np.arange(1,N+1):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout) 

    return Yh, xa

def closed_loop_evol(N, x0, Wout, sigma_in, rho):
    xa = x0.copy()
    Yh = np.empty((N, dim))    
    xat = np.empty((N, N_units))
    xat[0] = xa[:N_units]
    Yh[0] = np.dot(xa, Wout)
    
    for i in np.arange(1, N_transient):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)
        xat[i] = xa[:N_units].copy()

    for i in np.arange(N_transient, N):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout) 
        xat[i] = xa[:N_units].copy()
        
    return Yh, xat

# %% Define Margazoglou's ESNs

# load Margazoglou's parameter files
parameters = loadmat('ESN_RVC_Noise_dt_0.005_noise_0.0005_ens_2')
Win = parameters['Win']
W = parameters['W']
Wout = parameters['Wout']
norm = parameters['norm'][0]
fix_hyp = parameters['fix_hyp'][0]
opt_hyp = parameters['opt_hyp'][0]

# define parameters for ESN construction -- ensemble 1
dim = 3
N_units = 100

ld = opt_hyp[2].copy()
sigma_in = opt_hyp[1].copy()
rho = opt_hyp[0].copy()
C = parameters['Win'][0][0].toarray()
A = parameters['W'][0][0].toarray()
weights = parameters['Wout'][0, :, :]
bias_in = np.array([1.])
bias_out  = np.array([1.]) 
x0 = np.zeros(shape=(N_units,))

# %% define parameters for ESN construction -- ensemble 2

opt_hyp = parameters['opt_hyp'][1]

ld = opt_hyp[2].copy()
sigma_in = opt_hyp[1].copy()
rho = opt_hyp[0].copy()
C = parameters['Win'][0][1].toarray()
A = parameters['W'][0][1].toarray()
weights = parameters['Wout'][1, :, :]
bias_in = np.array([1.])
bias_out  = np.array([1.]) 
x0 = np.zeros(shape=(N_units,))

# %% evolve over time 
N_lyap = 222
N_washout = 2 * N_lyap
N_train = 200 * N_lyap
N_tstart = N_washout + N_train + 10
N_intt = 200 * N_lyap
N_transient = int((N_intt-1)/10)

i = 1

U_wash = U[N_tstart - N_washout + i*N_intt : N_tstart + i*N_intt].copy()       
Xa1 = open_loop(U_wash, x0, sigma_in, rho)
Uh_wash = np.dot(Xa1, weights) 
Yh_t, xat = closed_loop_evol(N_intt-1, Xa1[-1], weights, sigma_in, rho)

# %% compute Lyapunov spectrum using their jacobians

x_data = xat
z_data = Yh_t
esn_data = [(x_data[t], z_data[t]) for t in range(len(x_data))]

def const_jacobian(Wout):    
    dfdu = np.r_[np.diag(sigma_in/norm), [np.zeros(dim)]]
    d    = C.dot(dfdu) 
    c    = np.matmul(d, weights[:N_units, :].T) 
    return c, A.dot(np.diag(np.ones(N_units)*rho))

def jacobian_esn_func(t, esn_inputs):
    xa, z = esn_inputs
    const_jac_a, const_jac_b = const_jacobian(weights)
    diag_mat = np.diag(1 - xa[:N_units]*xa[:N_units])
    jacobian = np.matmul(diag_mat,const_jac_a) + np.matmul(diag_mat,const_jac_b)
    return jacobian

lorenz_esn_spectrum = lyaps_spec.lyapunov_spectrum(esn_data, N_units, 0.005, "difference", N_transient, jacobian_esn_func)

# %% compute Lyapunov spectrum for Lorenz

steps_trans_lor = int(20/0.005)

def jacobian_lorenz(t, lor_inputs):
    u, v, w = lor_inputs
    sig, beta, rho = 10, 8/3, 28
    
    dF1dinputs = [-sig, sig, 0]
    dF2dinputs = [rho-w, -1, -u]
    dF3dinputs = [v, u, -beta]
    
    return np.array([dF1dinputs, dF2dinputs, dF3dinputs])

lorenz_spectrum = lyaps_spec.lyapunov_spectrum(U, dim, 0.005, "ordinary", steps_trans_lor, jacobian_lorenz)

# %% Kaplan-Yorke dimension

lor_spectrum_sorted = np.sort(lorenz_spectrum)[::-1]
esn_spectrum_sorted = np.sort(lorenz_esn_spectrum)[::-1]

KY_Lor = lyaps_spec.kaplan_yorke(lor_spectrum_sorted)
KY_esn = lyaps_spec.kaplan_yorke(esn_spectrum_sorted)

print(KY_Lor, KY_esn)

# %% compare distributions
Y_t = U[N_tstart + i*N_intt : N_tstart + i*N_intt + N_intt].copy()
Yh_t_short = Yh_t[:6651]

u_actuals = Y_t[0]
v_actuals = Yh_t_short[0]

esn_help.hist_accuracy_plot(u_actuals, v_actuals, 'hist_marg.pdf')

