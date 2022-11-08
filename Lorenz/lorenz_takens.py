import numpy as np
import numdifftools as nd
from scipy.integrate import odeint

def lorenz(U, t, sigma, beta, rho):
    # The Lorenz equations.
    u, v, w = U
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

sigma, beta, rho = 10, 8/3, 28
u0, v0, w0 = 0, 1, 1.05

total_time = 3000
h = 0.02

t = np.arange(0, total_time, h)

solution = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
solution = 0.01*solution

N = 3

def phi(z_prev):    
    return z_prev + np.multiply(h, lorenz(z_prev, 0, sigma, beta, rho))

def omega(x):
    return x[0]

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

z_prev = solution[-2]
z_curr = solution[-1]
jacobian_Phi_z_prev = jacobian_Phi(z_prev)
jacobian_Phi_z_curr = jacobian_Phi(z_curr)

jacobian_phi = nd.Jacobian(phi, method='central', step=1e-6)(z_prev)
jacobian_F = jacobian_Phi_z_curr @ jacobian_phi @ np.linalg.pinv(jacobian_Phi_z_prev)

J_F_eigs = np.sort(np.linalg.eigvals(jacobian_F))[::-1]
J_phi_eigs = np.sort(np.linalg.eigvals(jacobian_phi))[::-1]
