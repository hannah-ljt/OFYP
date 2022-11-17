# sample code 
 
import cmath
import numpy as np
import scipy

def log(x):
    # helper function apply complex log to array element-wise
    # args: x - an array of complex or real scalars. 
    return np.array([cmath.log(xx) for xx in x])

def iter_rk45(prev, t, h, f, fargs=None):
    # helper function one iteration of rk45
    # args: prev - output from previous time step. 
          # t - time, if required in f. If not used in f, any value will do.
          # h - the size of time step used to generate next data point
          # f - callable with format def f(t, inputs, additional args)
              # additional args is default None. 
    
    if fargs == None:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1)
        z3 = prev + (h/2)*f(t + 0.5*h, z2)
        z4 = prev + h*f(t + 0.5*h, z3)

        z = (h/6)*(f(t, z1) + 2*f(t + 0.5*h, z2) + 2*f(t + 0.5*h, z3) + f(t + h, z4))
        curr = prev + z
    
    else:
        z1 = prev
        z2 = prev + (h/2)*f(t, z1, fargs)
        z3 = prev + (h/2)*f(t + 0.5*h, z2, fargs)
        z4 = prev + h*f(t + 0.5*h, z3, fargs)

        z = (h/6)*(f(t, z1, fargs) + 2*f(t + 0.5*h, z2, fargs) + 2*f(t + 0.5*h, z3, fargs) + f(t + h, z4, fargs))
        curr = prev + z
    
    return curr

# generate full lyapunov spectrum
def lyapunov_spectrum(data, N, h, eq_type, t_trans, jacobian, delta=10**(-7), seed=None):
    # compute lyapunov spectrum from data and known jacobian function 
    # Uses Bennetin's algorithm (see Appendix A of https://link.springer.com/book/10.1007/978-3-642-14938-2)
    # args: data - can be single dimensional or 2-dimensional but data should 
                 # be stored row-wise so that data[t] is the t-th data point.
          # N - dimension of system and also the number of exponents found
          # h - time step size that was used to generate dataset
          # eq_type - either differential (e.g. for lorenz, mackey-glass) 
                    # or difference (e.g. for esn)
          # t_trans - initial transient period before summing for final average
          # jacobian - callable that takes in (t, inputs)
          # delta - size of initial perturbation vector. Should be small.
                    
    if seed != None:
        np.random.seed(seed)
        Delta = delta * scipy.linalg.orth(np.random.rand(N, N))
    else:
        Delta = delta * np.identity(N)
    
    vec_len_sum = np.zeros(shape=(N, ), dtype='float32')

    for t in range(len(data)):
        data_t = data[t]

        if t % 100 == 0:
            if t <= t_trans:
                total_time = (t + 1) * h
            else:
                total_time = (t - t_trans + 1) * h
            lyapunov_spec = np.real(vec_len_sum) / total_time
            print(t, ": ", lyapunov_spec[0:10])

        if eq_type == "ordinary":
            for j in range(N):
                delta_j = lambda t, dy : jacobian(t, data_t) @ dy
                Delta[:, j] = iter_rk45(Delta[:, j], t, h, delta_j)
                
        if eq_type == "difference":
            Delta = jacobian(t, data_t) @ Delta

        Delta, R = np.linalg.qr(Delta)
        evolved_vec_lengths = np.diagonal(R)
        if t >= t_trans:
            vec_len_sum = vec_len_sum + log(evolved_vec_lengths)
    
    total_time = (len(data)-t_trans) * h
    lyapunov_spec = np.real(vec_len_sum) / total_time
    
    return lyapunov_spec

def lyapunov_spectrum_mj(data, N, h, t_trans, J_Delta, jacobian0, delta=10**(-7), seed=None):
    
    if seed != None:
        np.random.seed(seed)
        Delta_t_2 = delta * scipy.linalg.orth(np.random.rand(N, N))
        Delta_t_1 = jacobian0(0, data[0]) @ Delta_t_2
        Delta_t_1 = np.linalg.qr(Delta_t_1)[0]
    
    else:
        Delta_t_2 = delta * np.identity(N)
        Delta_t_1 = jacobian0(0, data[0]) @ Delta_t_2
        Delta_t_1 = np.linalg.qr(Delta_t_1)[0]
    
    vec_len_sum = np.zeros(shape=(N, ), dtype='float32')
    
    for t in range(1, len(data)):
    
        if t % 100 == 0:
            if t <= t_trans:
                total_time = (t + 1) * h
            else:
                total_time = (t - t_trans + 1) * h
                
            lyapunov_spec = np.real(vec_len_sum) / total_time
            print(t, ": ", lyapunov_spec[0:10])
        
        Delta_t = J_Delta(t, data[t], Delta_t_1, Delta_t_2) 
        
        Delta_t, R = np.linalg.qr(Delta_t)
        evolved_vec_lengths = np.diagonal(R)
        
        if t >= t_trans:
            vec_len_sum = vec_len_sum + log(evolved_vec_lengths)
            
        Delta_t_2 = Delta_t_1
        Delta_t_1 = Delta_t
    
    total_time = (len(data)-t_trans) * h
    lyapunov_spec = np.real(vec_len_sum) / total_time
    
    return lyapunov_spec


def kaplan_yorke(sorted_array):
    
    exp_sum = 0
    D_KY = 0

    for idx in range(0, len(sorted_array)):
        exp_sum = exp_sum + sorted_array[idx]
        if exp_sum < 0:
            D_KY = idx - 1 - ((exp_sum - sorted_array[idx])/sorted_array[idx])
            break
            
    D_KY = D_KY + 1
    
    return D_KY
        
        