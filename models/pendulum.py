"""
This module defines a class solving for the motion
for a single pendulum system, with and without damping.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ODEsNotSolved(Exception):
    """Failed attempt at accessing solutions.
    A call to the ODE systems solve method must be
    carried out before the solution properties
    can be used.
    """
    pass


class Pendulum:
    """Class for representing a single pendulum system.
    A pendulum of mass M is assumed to be hanging from a
    massless, stiff rod of length L. The pendulum
    is assumed to swing freely, only affected by gravity,
    i.e., no friction or air resistance.
    """

    def __init__(self, M=1, L=1, G=9.81):
        """Define the physical parameters of the system.
        Parameters
        ----------
        M : float, optional
            Mass of the pendulum in kg, defaults to 1
        L : float, optional
            Length of the pendulum in m, defaults to 1
        G : float, optional
            Acceleration of gravity felt by the pendulum
            in m/s/s. Defaults to 9.81.
        """
        self.M = M
        self.L = L
        self.G = G

    def __call__(self, t, y):
        """RHS of the ODE system.
        Parameters
        ----------
        t : float
            The time point
        y : tuple of floats
            A tuple of the pendulums states at time t,
            y = (theta, omega)
        """
        print(f"{t=}", type(t))
        theta, omega = y
        L = self.L
        G = self.G
        return (omega, -G / L * np.sin(theta))

    def solve(self, y0, T, dt, angles='rad', **kwargs):
        """Solve for the motion of the system.
        The equations of motions are solved on the interval (0, T]
        and the solutions evaluted at a given interval. The solutions
        are not returned, but stored as class attributes.

        If multiple calls to solve are made, they are treated
        independently, with the newest one overwriting any old
        solution data.

        Notes
        -----
        The ODEs are solved numerically using the function
        scipy.integrate.solve_ivp.
        Parameters
        ----------
        y0 : tuple
            The initial condition of the system: (theta, omega)
        T : float
            End time
        dt : float
            Time step where solutions are evaluated
        angles : string, optional
            Whether initial conditions are in radians or degrees
            should be either 'rad' (default) or 'deg'
        **kwargs
            Arbitrary keyword arguments are passed along
            to scipy.integrate.solve_ivp.
        """
        if angles == 'deg':
            y0 = np.radians(y0)

        t = np.arange(0, T + dt, dt)
        solution = solve_ivp(self, (0, T), y0, t_eval=t, **kwargs)
        self._time = solution.t
        self._theta = solution.y[0]
        self._omega = solution.y[1]

    @property
    def t(self):
        """Array of time points."""
        try:
            return self._time
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def theta(self):
        """Array of the angular displacement of the pendulum."""
        try:
            return self._theta
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def omega(self):
        """Array of the angular velocity of the pendulum."""
        try:
            return self._omega
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def x(self):
        """Array of x-position of the pendulum over time."""
        return self.L * np.sin(self.theta)

    @property
    def y(self):
        """Array of y-position of the pendulum over time."""
        return -self.L * np.cos(self.theta)

    @property
    def vx(self):
        """Array of linear velocity in x-direction."""
        return np.gradient(self.x, self.t)

    @property
    def vy(self):
        """Array of linear velocity in y-direction."""
        return np.gradient(self.y, self.t)

    @property
    def kinetic(self):
        """Array of the kinetic energy over time."""
        return 0.5 * self.M * (self.vx**2 + self.vy**2)

    @property
    def potential(self):
        """Array of the potential energy over time."""
        return self.M * self.G * (self.y + self.L)

    @property
    def total_energy(self):
        """Array of the total mechanical energy over time."""
        return self.kinetic + self.potential


class DampenedPendulum(Pendulum):
    """Class for representing a dampened pendulum system.
    This class subclasses the single pendulum class
    to include damping of the pendulum due to
    air resistance and friction.
    The damping is assumed to be linear in the
    pendulums angular velocity.
    """

    def __init__(self, B, M=1, L=1, G=9.81):
        """Define the physical parameters of the system.
        Parameters
        ----------
        B : float
            The damping parameter B, given in units
            of kg/s.
        M : float, optional
            Mass of the pendulum in kg, defaults to 1
        L : float, optional
            Length of the pendulum in m, defaults to 1
        G : float, optional
            Acceleration of gravity felt by the pendulum
            in m/s/s. Defaults to 9.81.
        """
        self.B = B
        self.M = M
        self.L = L
        self.G = G

    def __call__(self, t, y):
        """RHS of the ODE system.
        Parameters
        ----------
        t : float
            The time point
        y : tuple of floats
            A tuple of the pendulums states at time t,
            y = (theta, omega)
        """
        theta, omega = y
        B = self.B
        M = self.M
        L = self.L
        G = self.G
        return (omega, -G / L * np.sin(theta) - B / M * omega)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Pendulum()
    T = 10
    dt = 0.5
    model.solve((60, 0), T, dt, angles='deg', method='Radau')

    plt.plot(model.t, model.theta)
    plt.xlabel('Time [seconds]')
    plt.ylabel('Angular displacement [radians]')
    plt.show()

    '''
    plt.plot(model.t, model.kinetic, label='Kinetic')
    plt.plot(model.t, model.potential, label='Potential')
    plt.plot(model.t, model.total_energy, label='Total')
    plt.legend()
    plt.xlabel('Time [seconds]')
    plt.title('Mechanical Energy of the Pendulum over time')
    plt.show()

    model2 = DampenedPendulum(0.25)
    model2.solve((60, 0), 20, 0.001, angles='deg', method='Radau')
    plt.plot(model.t, model.theta, label='Not Dampened')
    plt.plot(model2.t, model2.theta, label='Dampened')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Angular displacement [radians]')
    plt.show()

    plt.plot(model2.t, model2.kinetic, label='Kinetic')
    plt.plot(model2.t, model2.potential, label='Potential')
    plt.plot(model2.t, model2.total_energy, label='Total')
    plt.legend()
    plt.xlabel('Time [seconds]')
    plt.title('Mechanical Energy of the Dampened Pendulum over time')
    plt.show()
    '''
