

class HodgkinHuxley:

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

        # This works:
        # constant_I = 10
        # dVdt = constant_I - foo(V, n, m, h)

        # This doesn't:
        dVdt = self.I(t) - foo(V, n, m, h)

        dndt = foo(V, n)
        dmdt = foo(V, m)
        dhdt = foo(V, h)
        return [dVdt, dndt, dmdt, dhdt]

    @property
    def _initial_conditions(self):
        n0 = self.n_inf(self.V_rest)
        m0 = self.m_inf(self.V_rest)
        h0 = self.h_inf(self.V_rest)
        return (self.V_rest, n0, m0, h0)

    def solve(self, stimulus, T, dt, y0=None, **kwargs):

        if y0 is None:
            y0 = self._initial_conditions

        t_eval = np.arange(0, T + dt, dt)

        # Interpolate stimulus
        self.I = interp1d(x=t_eval, y=stimulus)  # linear spline

        # Stimulus as array
        #self.I = stimulus

        solution = solve_ivp(self, t_span=(0, T), y0=y0,
                             t_eval=t_eval, **kwargs)

        self._time = solution.t
        self._V = solution.y[0]
        self._n = solution.y[1]
        self._m = solution.y[2]
        self._h = solution.y[3]
