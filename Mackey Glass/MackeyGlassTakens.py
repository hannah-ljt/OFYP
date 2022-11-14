import numpy as np
import numdifftools as nd

def iter_rk45(x_prev, x_lag, h, f, fargs):   
    
    z1 = x_prev
    z2 = x_prev + (h/2)*f(z1, x_lag, fargs)
    z3 = x_prev + (h/2)*f(z2, x_lag, fargs)
    z4 = x_prev + h*f(z3, x_lag, fargs)
    
    z = (h/6)*(f(z1, x_lag, fargs) + 2*f(z2, x_lag, fargs) + 2*f(z3, x_lag, fargs) + f(z4, x_lag, fargs))
    x_curr = x_prev + z
    
    return x_curr

def dde_rk45(n_intervals, init, f, fargs, discretisation=50):
    
    delay = fargs['delay']
    h = delay / (discretisation - 1)   # step-size

    prev_array = [ init(t) for t in range(discretisation) ]
    curr_array = [0] * discretisation
    solution_array = []
    
    for interval in range(0, n_intervals):
        curr_array[0] = iter_rk45(prev_array[discretisation-1], prev_array[0], h, f, fargs)
        for step in range(1, discretisation):
            curr_array[step] = iter_rk45(curr_array[step-1], prev_array[step], h, f, fargs)
        solution_array.append(curr_array.copy())
        prev_array = curr_array
    return np.array(solution_array)

mg_args = {'delay': 17,
           'a': 0.2, 
           'b': 0.1, 
           'n': 10}

def mackeyglass(z, z_lag, mg_args):
    
    a = mg_args['a']
    b = mg_args['b']
    n = mg_args['n']
    
    return (a * z_lag) / (1 + z_lag**n) - b * z

def init(t):
    return 1.2

delay = mg_args['delay']
N = 50
Delta_t = delay / (N - 1)

data = dde_rk45(1000, init, mackeyglass, mg_args, discretisation=N)

def phi(z_prev):
    z_curr = np.empty(shape=(N, ))
    z_curr[0] = z_prev[N-1] + Delta_t * mackeyglass(z_prev[N-1], z_prev[0], mg_args)
    for i in range(1, N):
        z_curr[i] = z_curr[i-1] + Delta_t * mackeyglass(z_curr[i-1], z_prev[i], mg_args)
    return z_curr

def omega(vec):
    return vec[0]

def phi_power_k(z_arg, k_arg):
    if k_arg == 0:
        return z_arg
    return phi(phi_power_k(z_arg, k_arg-1))

def jacobian_Phi(z):
    J_Phi = np.zeros(shape=(2*N+1, N))
    for k in range(0,2*N+1):
        def phi_power(z_arg):
            return phi_power_k(z_arg, k)
        jacobian_phi_power = nd.Jacobian(phi_power, method='central', step=1e-6)(z)
        inverse = np.linalg.inv(jacobian_phi_power)
        jacobian_Phi_row = nd.Jacobian(omega)(z) @ inverse
        J_Phi[2*N - k] = jacobian_Phi_row 
    return J_Phi 

z_prev = data[998]
z_curr = data[999]
jacobian_Phi_z_prev = jacobian_Phi(z_prev)
jacobian_Phi_z_curr = jacobian_Phi(z_curr)

jacobian_phi = nd.Jacobian(phi, method='central', step=1e-6)(z_prev)

regularisation = 10**(-15)
jacobian_F = jacobian_Phi_z_curr @ jacobian_phi @ np.linalg.pinv(jacobian_Phi_z_prev)

J_F_eigs = np.sort(np.linalg.eigvals(jacobian_F))[::-1]
J_phi_eigs = np.sort(np.linalg.eigvals(jacobian_phi))[::-1]