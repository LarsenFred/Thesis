import numpy as np
from scipy import interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import tools

def util(c,par):
    return (c**(1-par.rho)-1)/(1-par.rho)

def marg_util(c,par):
    return c**(-par.rho)

def inv_marg_util(u,par):
    return u**(-1/par.rho)


def setup():
    class par: pass
    par.beta = 0.98
    par.rho = 0.5
    par.R = 1/par.beta
    par.sigma = 0.02
    par.mu = 0
    par.M = 10
    par.T = 10

    # Gauss Hermite weights and points
    par.num_shocks = 5
    x,w = gauss_hermite(par.num_shocks)
    par.eps = np.exp(par.sigma*np.sqrt(2)*x)
    par.eps_w = w/np.sqrt(np.pi)

    # simulation parameters
    par.simN = 10000
    par.M_ini = 1.5

    # Grid
    par.num_a = 100
    #4. end of period assets
    par.grid_a = nonlinspace(0 + 1e-8,par.M,par.num_a,1.1)

    #dimension of value function space
    par.dim = [par.num_a,par.T]

    return par

def EGM_loop (sol,t,par):
    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value="extrapolate")
    for i_a,a in enumerate(par.grid_a): #loop over end og period assets
        #future m and c
        m_next = par.R * a + par.eps
        c_next = interp(m_next)

        #future expected marginal utility
        EU_next = np.sum(par.eps_w * marg_util(c_next,par))

        #current consumption from FOC
        c_now = inv_marg_util(par.R * par.beta * EU_next, par)

        # index 0 is used for the corner solution so start at index 1
        sol.C[i_a+1,t] = c_now
        sol.M[i_a+1,t] = a + c_now
    return sol

def EGM_vectorized (sol, t, par):

    interp = interpolate.interp1d(sol.M[:,t+1],sol.C[:,t+1], bounds_error=False, fill_value="extrapolate")
    
    # future m and c
    m_next = par.R * par.grid_a[:,np.newaxis] + par.eps[np.newaxis,:] # next period assets
    c_next = interp(m_next) # next period consumption

    # future expected marginal utility
    EU_next = np.sum(par.eps_w[np.newaxis,:] * marg_util(c_next,par), axis=1)

    # current consumption
    c_now = inv_marg_util(par.R * par.beta * EU_next, par)

    # index 0 is used for the corner solution so start at index 1
    sol.C[1:,t] = c_now
    sol.M[1:,t] = par.grid_a + c_now
    return sol

def solve_EGM(par, vector=False):
    #initialize solution class 
    class sol: pass
    shape = [par.num_a+1, par.T]
    sol.C = np.nan + np.zeros(shape)
    sol.M = np.nan + np.zeros(shape)
    # last period: consume everything
    sol.M[:,par.T-1]=nonlinspace(0,par.M,par.num_a+1,1.1)
    sol.C[:,par.T-1]=sol.M[:,par.T-1].copy()

    # loop over periods
    for t in range(par.T-2, -1, -1):
        if vector == True:
            sol = EGM_vectorized(sol, t, par)
        else:
            sol = EGM_loop(sol, t, par)
    return sol


def gauss_hermite(n):

    # a. calculations
    i = np.arange(1, n)
    a = np.sqrt(i / 2)
    CM = np.diag(a, 1) + np.diag(a, -1)
    L, V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:, I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(np.pi) * V[:, 0]**2

    return x, w

def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between unequal spacing
    phi = 1 -> equal spacing
    phi up -> more points closer to minimum
    """
    assert x_max > x_min
    assert n >= 2
    assert phi >= 1

    # 1. recursion
    y = np.empty(n)

    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi

    # 3. assert increasing
    assert np.all(np.diff(y) > 0)

    return y

def simulate (par,sol):
    
    # Initialize
    class sim: pass
    dim = (par.simN, par.T)
    sim.M = par.M_ini*np.ones(dim)
    sim.C = np.nan + np.zeros(dim)
    np.random.seed(2022)

    # Simulate
    for t in range(par.T):
        interp = interpolate.interp1d(sol.M[:,t], sol.C[:,t], bounds_error=False, fill_value="extrapolate")
        sim.C[:,t] = interp(sim.M[:,t]) # find consumption given state

        if t<par.T-1: #if not in last period
            logY = np.random.normal(par.mu,par.sigma,par.simN) # draw random number from normal dist
            Y = np.exp(logY)
            A = sim.M[:,t] - sim.C[:,t]

            sim.M[:,t+1] = par.R * A + Y # the state in the following period

    return sim



# solve model using EGM model
par_EGM = setup()


sol_EGM = solve_EGM(par_EGM, vector=False)


# Print consumption function
fig = plt.figure(figsize=(8,5))# figsize is in inches...
ax = fig.add_subplot(1,1,1)
for t in range(par_EGM.T):
    ax.plot(sol_EGM.M[:,t],sol_EGM.C[:,t],  label=f"t = {t + 1}, EGM",  linestyle='-')
ax.set_xlabel(f"$M_t$")
ax.set_ylabel(f"$C_t$")
ax.set_title(f'Consumption function')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


sim = simulate(par_EGM,sol_EGM)

def figure_sim(simC,par):
    fig = plt.figure(figsize=(10,4))# figsize is in inches...
    t_grid = [t for t in range(1,par.T+1)]    
    ax = fig.add_subplot(1,1,1)
    ax.plot(t_grid,np.mean(sim.C,0),'-o')
    ax.set_xlabel(f"t")
    ax.set_ylabel(f"Mean of consumption")
    ax.set_ylim(bottom=0.5,top=1.5)
    plt.show()
    
figure_sim(sim.C,par_EGM)