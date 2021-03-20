#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import odeint, solve_ivp

'''
Scipy note:
For new code, use scipy.integrate.solve_ivp to solve a differential equation.

scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
                          events=None, vectorized=False, args=None, **options)
'''


class HodgkinHuxley:
    """
    The Hodgkin–Huxley model describes how action potentials in neurons are
    initiated and propagated. From a biophysical point of view, action
    potentials are the result of currents that pass through ion channels in
    the cell membrane. In an extensive series of experiments on the giant axon
    of the squid, Hodgkin and Huxley succeeded to measure these currents and
    to describe their dynamics in terms of differential equations.

    Arguments
    ---------
    dt : float, default 0.025
        Time step in ms
    T :
        Simulation time in ms
    I : {int, float, str, callable}, default 150
        Input current (stimulus) in μA/cm**2

    """

    def __init__(self, dt=0.025, T=50.0, seed=None):

        # Hodgkin-Huxley model parameters
        self.V_rest = -65        # mV
        self.Cm = 1              # μF/cm**2
        self.gbar_Na_true = 120  # mS/cm**2
        self.gbar_K_true = 36    # mS/cm**2
        self.gbar_L = 0.3        # mS/cm**2
        self.E_Na = 50           # mV
        self.E_K = -77           # mV
        self.E_L = -54.4         # mV

        # setup parameters and state variables
        '''
        if isinstance(I, (int, float)):
            self.I_value = I
        '''

        self.time = np.arange(0, T + dt, dt)

        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def __call__(self, gbar_K, gbar_Na):
        pass

    # stimulus
    '''
    def I(self, t):
        return 150
    '''

    def I(self, t):
        t_stim_on = 5
        t_stim_off = 30
        if t_stim_on <= t <= t_stim_off:
            I = 10
        else:
            I = 0
        return I

    # K channel kinetics

    def alpha_n(self, V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10.))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80.)

    # Na channel kinetics (activating)
    def alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10.))

    def beta_m(self, V):
        return 4 * np.exp(-(V + 65) / 18.)

    # Na channel kinetics (inactivating)
    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65) / 20.)

    def beta_h(self, V):
        return 1. / (1 + np.exp(-(V + 35) / 10.))

    # steady-states and time constants
    def n_inf(self, V):
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def tau_n(self, V):
        return 1. / (self.alpha_n(V) + self.alpha_n(V))

    def m_inf(self, V):
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def tau_m(self, V):
        return 1. / (self.alpha_m(V) + self.alpha_m(V))

    def h_inf(self, V):
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

    def tau_h(self, V):
        return 1. / (self.alpha_h(V) + self.alpha_h(V))

    # Hodgkin-Huxley ODEs
    def dndt(self, n, V):
        return self.alpha_n(V) * (1 - n) - self.beta_n(V) * n

    def dmdt(self, m, V):
        return self.alpha_m(V) * (1 - m) - self.beta_m(V) * m

    def dhdt(self, h, V):
        return self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

    def dVdt(self, X, t, gbar_K, gbar_Na):
        V, n, m, h = X

        g_K = gbar_K * (n**4)
        g_Na = gbar_Na * (m**3) * h
        g_l = self.gbar_L

        dndt = self.dndt(n, V)
        dmdt = self.dmdt(m, V)
        dhdt = self.dhdt(h, V)

        dVdt = (self.I(t) - g_Na * (V - self.E_Na) - g_K *
                (V - self.E_K) - g_l * (V - self.E_L)) / self.Cm

        return [dVdt, dndt, dmdt, dhdt]

    # initial conditions
    @property
    def initial_conditions(self):
        n0 = self.n_inf(self.V_rest)
        m0 = self.m_inf(self.V_rest)
        h0 = self.h_inf(self.V_rest)
        return [self.V_rest, n0, m0, h0]

    # simulator
    def simulator(self, gbar_K, gbar_Na):
        X = odeint(self.dVdt, y0=self.initial_conditions,
                   t=self.time, args=(gbar_K, gbar_Na))
        V = X[:, 0]
        return self.time, V

    def observed_data(self, noise=None):
        # self.gbar_Na = 120  # mS/cm**2
        # self.gbar_K = 36    # mS/cm**2
        pass

    def plot_observed_data(self):
        pass


def HH_simulator():
    """HH wrapper that returns set of summary statistics"""
    hh = HodgkinHuxley()


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    gbar_K = 36    # mS/cm**2
    gbar_Na = 120  # mS/cm**2

    hh = HodgkinHuxley()
    t, V = hh.simulator(gbar_K, gbar_Na)

    plt.plot(t, V)
    plt.show()

    '''
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    def func(t, x, u):
        dydt = (-x + u(t)) / 5
        return dydt

    y0 = 0
    t_span = [0, 10]
    t_eval = np.linspace(0, 10, 1000)
    u_value = np.random.uniform(-1, 1, 1000)
    u = interp1d(x=t_eval, y=u_value)

    sol = solve_ivp(func, t_span=t, y0=y0, t_eval=t_eval, args=(u, ))
    '''
