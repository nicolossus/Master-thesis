#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ignore overflow warnings; occurs with certain np.exp() evaluations
np.warnings.filterwarnings('ignore', 'overflow')


class ODEsNotSolved(Exception):
    """Failed attempt at accessing solutions.

    A call to the ODE systems solve method must be
    carried out before the solution properties
    can be used.
    """
    pass


class HodgkinHuxley:
    """Class for representing the Hodgkin-Huxley model.

    The Hodgkin–Huxley model describes how action potentials in neurons are
    initiated and propagated. From a biophysical point of view, action
    potentials are the result of currents that pass through ion channels in
    the cell membrane. In an extensive series of experiments on the giant axon
    of the squid, Hodgkin and Huxley succeeded to measure these currents and
    to describe their dynamics in terms of differential equations.

    All model parameters can be accessed (get or set) as class attributes.

    The following solutions are available as class attributes after calling
    the class method `solve`:

    Attributes
    ----------
    t : array_like
        The time array of the spike.
    V : array_like
        The voltage array of the spike.
    Vm : array_like
        The voltage array of the spike (alias for V).
    n : array_like
        The state variable of the potassium channel.
    m : array_like
        The activating state variable of the sodium channel.
    h : array_like
        The inactivating state variable of the sodium channel.
    """

    def __init__(self, V_rest=-65., Cm=1., gbar_K=36., gbar_Na=120., gbar_L=0.3, E_K=-77., E_Na=50., E_L=-54.4):
        """
        Define the model parameters.

        Arguments
        ---------
        V_rest : float, default: -65.
            Resting potential of neuron in units: mV
        Cm : float, default: 1.
            Membrane capacitance in units: μF/cm**2
        gbar_K : float, default: 36.
            Potassium conductance in units: mS/cm**2
        gbar_Na : float, default: 120.
            Sodium conductance in units: mS/cm**2
        gbar_L : float, default: 0.3.
            Leak conductance in units: mS/cm**2
        E_K : float, default: -77.
            Potassium reversal potential in units: mV
        E_Na : float, default: 50.
            Sodium reversal potential in units: mV
        E_L : float, default: -54.4
            Leak reversal potential in units: mV

        Notes
        -----
        Default parameter values as given by Hodgkin and Huxley (1952).

        References
        ----------
        Hodgkin, A. L., Huxley, A.F. (1952).
        "A quantitative description of membrane current and its application
        to conduction and excitation in nerve".
        J. Physiol. 117, 500-544.
        """

        # Hodgkin-Huxley model parameters
        self._V_rest = V_rest      # resting potential [mV]
        self._Cm = Cm              # membrane capacitance [μF/cm**2]
        self._gbar_K = gbar_K      # potassium conductance [mS/cm**2]
        self._gbar_Na = gbar_Na    # sodium conductance [mS/cm**2]
        self._gbar_L = gbar_L      # leak coductance [mS/cm**2]
        self._E_K = E_K            # potassium reversal potential [mV]
        self._E_Na = E_Na          # sodium reversal potential [mV]
        self._E_L = E_L            # leak reversal potential [mV]

    def __call__(self, t, y):
        """RHS of the Hodgkin-Huxley ODEs.

        Parameters
        ----------
        t : float
            The time point
        y : tuple of floats
            A tuple of the state variables, y = (V, n, m, h)
        """

        V, n, m, h = y
        dVdt = (self.I(t) - self._gbar_K * (n**4) * (V - self._E_K) - self._gbar_Na *
                (m**3) * h * (V - self._E_Na) - self._gbar_L * (V - self._E_L)) / self._Cm
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        return [dVdt, dndt, dmdt, dhdt]

    # K channel kinetics
    def alpha_n(self, V):
        return 0.01 * (V + 55.) / (1 - np.exp(-(V + 55.) / 10.))

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

    @property
    def _initial_conditions(self):
        """Default Hodgkin-Huxley model initial conditions."""
        n0 = self.n_inf(self.V_rest)
        m0 = self.m_inf(self.V_rest)
        h0 = self.h_inf(self.V_rest)
        return (self.V_rest, n0, m0, h0)

    def solve(self, stimulus, T, dt, y0=None, **kwargs):
        """Solve the Hodgkin-Huxley equations.

        The equations are solved on the interval (0, T] and the solutions
        evaluted at a given interval. The solutions are not returned, but
        stored as class attributes.

        If multiple calls to solve are made, they are treated independently,
        with the newest one overwriting any old solution data.

        Notes
        -----
        The ODEs are solved numerically using the function
        `scipy.integrate.solve_ivp`. For details, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        If `stimulus` is passed as an array, it and the time array, defined by
        `T` and `dt`, will be used to create an interpolation function via
        `scipy.interpolate.interp1d`.

        Credits to supervisor Joakim Sundnes for helping unravel the following.

        `solve_ivp` is an ODE solver with adaptive step size. If the keyword
        argument `first_step` is not specified, the solver will empirically
        select an initial "good" step size with the function `select_initial_step`
        (https://github.com/scipy/scipy/blob/master/scipy/integrate/_ivp/common.py#L64).
        This function calculates two proposals and returns the smallest. It
        first calculates an intermediate proposal, h0, that is based on the
        initial condition (y0) and the ODE's RHS evaluated for the initial
        condition (f0). For the standard Hodgkin-Huxley model, however, this
        estimated step size will be very large due to unfortunate circumstances
        (because norm(y0) > 0 while norm(f0) ~= 0). Since h0 only is an
        intermediate calculation, it is not used or returned by the solver.
        However, it is used to calculate the next proposal, h1, by calling the
        RHS. Normally, this procedure poses no problem, but can fail if an
        object with a limited interval is present in the RHS, such as an
        `interp1d` object.

        In the case of the standard Hodgkin-Huxley model, one might be tempted
        to pass the stimulus as an array to the solver. In order for `solve_ivp`
        to be able to evaluate the stimulus, it must be passed as an callable or
        constant. Thus, if an array is passed to the solver, an interpolation
        function must be created, in this implementation done with `interp1d`,
        for `solve_ivp` to be able to evaluate it. For the reasons explained
        above, the program will hence terminate unless the `first_step` keyword
        is specified and is set to a sufficiently small value. In this
        implementation, `first_step=dt` is already set in `solve_ivp`.

        The `solve_ivp` keyword `max_step` should be considered to be specified
        for stimuli over short time spans, in order to ensure that the solver
        does not step over them.

        Note that `first_step` still needs to specified even if `max_step` is.
        `select_initial_step` will be called regardless if `first_step` is not
        specified, and the calls for calculating h1 will be done before checking
        whether h0 is larger than than the max allowed step size or not. Thus
        will only specifying `max_step` still result in program termination if
        `stimulus` is passed as an array. (Will not be a problem in this
        implementation since `first_step` is already specified.)

        Parameters
        ----------
        stimulus : array, shape (int(T/dt)+1,) or callable
            Input stimulus in units: μA/cm**2. If callable, the call signature
            must be '(t)'
        T : float
            End time in milliseconds (ms)
        dt : float
            Time step where solutions are evaluated
        y0 : array_like, shape (4,), default None
            Initial state of state variables V, n, m, h. If None, the default
            Hodgkin-Huxley model's initial conditions will be used.
        **kwargs
            Arbitrary keyword arguments are passed along to
            scipy.integrate.solve_ivp
        """

        # error-handling
        if not isinstance(dt, (int, float)):
            msg = (f"{dt=}".split('=')[0]
                   + " must be given as an int or float")
            raise TypeError(msg)

        if dt <= 0:
            msg = ("dt > 0 is required")
            raise ValueError(msg)

        if not isinstance(T, (int, float)):
            msg = (f"{T=}".split('=')[0]
                   + " must be given as an int or float")
            raise TypeError(msg)

        if T <= 0:
            msg = ("T > 0 is required")
            raise ValueError(msg)

        # times at which to store the computed solution
        t_eval = np.arange(0, T + dt, dt)

        if y0 is None:
            # use default HH initial conditions
            y0 = self._initial_conditions

        # handle the passed stimulus
        if callable(stimulus):
            self.I = stimulus
        elif isinstance(stimulus, np.ndarray):
            if not stimulus.shape == t_eval.shape:
                msg = ("stimulus numpy.ndarray must have shape (int(T/dt)+1)")
                raise ValueError(msg)
            # Interpolate stimulus
            self.I = interp1d(x=t_eval, y=stimulus)  # linear spline
        else:
            msg = ("'stimulus' must be either a callable function of t "
                   "or a numpy.ndarray of shape (int(T/dt)+1)")
            raise ValueError(msg)

        # solve HH ODEs
        solution = solve_ivp(self, t_span=(0, T), y0=y0,
                             t_eval=t_eval, first_step=dt, max_step=5 * dt, **kwargs)

        # store solutions
        self._time = solution.t
        self._V = solution.y[0]
        self._n = solution.y[1]
        self._m = solution.y[2]
        self._h = solution.y[3]

    # getters and setters
    @property
    def V_rest(self):
        """Get resting potential."""
        return self._V_rest

    @V_rest.setter
    def V_rest(self, V_rest):
        """Set resting potential."""
        if not isinstance(V_rest, (int, float)):
            msg = (f"{V_rest=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._V_rest = V_rest

    @property
    def Cm(self):
        """Get membrane capacitance."""
        return self._Cm

    @Cm.setter
    def Cm(self, Cm):
        """Set membrane capacitance."""
        if not isinstance(Cm, (int, float)):
            msg = (f"{Cm=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._Cm = Cm

    @property
    def gbar_K(self):
        """Get potassium conductance."""
        return self._gbar_K

    @gbar_K.setter
    def gbar_K(self, gbar_K):
        """Set potassium conductance."""
        if not isinstance(gbar_K, (int, float)):
            msg = (f"{gbar_K=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_K = gbar_K

    @property
    def gbar_Na(self):
        """Get sodium conductance."""
        return self._gbar_Na

    @gbar_Na.setter
    def gbar_Na(self, gbar_Na):
        """Set sodium conductance."""
        if not isinstance(gbar_Na, (int, float)):
            msg = (f"{gbar_Na=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_Na = gbar_Na

    @property
    def gbar_L(self):
        """Get leak conductance."""
        return self._gbar_L

    @gbar_L.setter
    def gbar_L(self, gbar_L):
        """Set leak conductance."""
        if not isinstance(gbar_L, (int, float)):
            msg = (f"{gbar_L=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_L = gbar_L

    @property
    def E_K(self):
        """Get potassium reversal potential."""
        return self._E_K

    @E_K.setter
    def E_K(self, E_K):
        """Set potassium reversal potential."""
        if not isinstance(E_K, (int, float)):
            msg = (f"{E_K=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_K = E_K

    @property
    def E_Na(self):
        """Get sodium reversal potential."""
        return self._E_Na

    @E_Na.setter
    def E_Na(self, E_Na):
        """Set sodium reversal potential."""
        if not isinstance(E_Na, (int, float)):
            msg = (f"{E_Na=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_Na = E_Na

    @property
    def E_L(self):
        """Get leak reversal potential."""
        return self._E_L

    @E_L.setter
    def E_L(self, E_L):
        """Set leak reversal potential."""
        if not isinstance(E_L, (int, float)):
            msg = (f"{E_L=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_L = E_L

    @ property
    def t(self):
        """Array of time points."""
        try:
            return self._time
        except AttributeError as e:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def V(self):
        """Values of the solution of V at t."""
        try:
            return self._V
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def Vm(self):
        """Values of the solution of V at t. (Alias for property V)."""
        try:
            return self._V
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def n(self):
        """Values of the solution of n at t."""
        try:
            return self._n
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def m(self):
        """Values of the solution of m at t."""
        try:
            return self._m
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def h(self):
        """Values of the solution of n at t."""
        try:
            return self._h
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from stimulus import constant_stimulus

    # plt.style.use('seaborn')
    sns.set()
    sns.set_context("paper")
    sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

    # Set fontsizes in figures
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'large',
              'axes.titlesize': 'large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'legend.fontsize': 'large',
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)

    # remove top and right axis from plots
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # simulation parameters
    T = 50.
    dt = 0.025
    I_amp = 0.32
    r_soma = 40
    threshold = -55  # AP threshold

    # input stimulus
    stimulus = constant_stimulus(
        I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=40, r_soma=r_soma)
    # print(stimulus["info"])
    I = stimulus["I"]
    I_stim = stimulus["I_stim"]

    # HH simulation
    hh = HodgkinHuxley()

    hh.V_rest = np.array([0, 1])

    hh.solve(I, T, [dt])
    t = hh.t
    V = hh.V
    n = hh.n
    m = hh.m
    h = hh.h

    # plot voltage trace
    fig = plt.figure(figsize=(8, 7), tight_layout=True, dpi=120)
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 1])
    ax = plt.subplot(gs[0])
    plt.plot(t, V, lw=2)
    plt.text(0.01, 0.25, 'Threshold', fontsize=12,
             color='darkred', transform=plt.gca().transAxes)
    plt.axhline(threshold, xmax=0.253, ls=':', color='darkred')
    plt.ylabel('Voltage (mV)')
    ax.set_xticks([])
    ax.set_yticks([-80, -55, -20, 10, 40])

    ax = plt.subplot(gs[1])
    plt.plot(t, n, '-.', lw=1.5, label='$n$')
    plt.plot(t, m, "--", lw=1.5, label='$m$')
    plt.plot(t, h, ls=':', lw=1.5, label='$h$')
    plt.legend(loc='upper right')
    plt.ylabel("State")
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax = plt.subplot(gs[2])
    plt.plot(t, I_stim, 'k', lw=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Stimulus (nA)')

    ax.set_xticks([0, 10, 25, 40, np.max(t)])
    ax.set_yticks([0, np.max(I_stim)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    fig.suptitle("Hodgkin-Huxley Model")
    plt.show()
