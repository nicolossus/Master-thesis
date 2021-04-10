#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
from pylfi.priors import ContinuousPrior, DiscretePrior


class Uniform(ContinuousPrior):
    r"""A uniform continuous random variable.

    Notes
    -----
    In the standard form, the distribution is uniform on ``[0, 1]``. Using the
    parameters ``loc`` and ``scale``, one obtains the uniform distribution on
    ``[loc, loc + scale]``.

    Examples
    --------
    >>> import numpy as np
    >>> from pylfi.priors import Uniform

    Initialize prior distribution for random variate :math:`\theta`:

    >>> theta_prior = Uniform(loc=0, scale=1, name='theta', tex=r'$\theta$', seed=42)

    Draw from prior:

    >>> theta = theta_prior.rvs(size=10)

    Evaluate probability density function:

    >>> x = np.linspace(-1, 2, 1000)
    >>> pdf = theta_prior.pdf(x)

    Display the probability density function:

    >>> theta_prior.plot_prior(x)
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        """
        Initialize distribution.

        Parameters
        ----------
        loc : array_like, optional
            Location parameter (default=0)
        scale: array_like, optional
            Scale parameter (default=1)
        name : str
            Name of random variate. Default is None (which will raise an ``Error``).
        tex : raw str literal, optional
            LaTeX formatted name of random variate given as ``r"foo"``. Default is
            None.
        rng : random number generator, optional
            Defines the random number generator to be used. Default is
            np.random.RandomState
        seed : {None, int}, optional
            This parameter defines the object to use for drawing random
            variates. If seed is None the RandomState singleton is used.
            If seed is an int, a new RandomState instance is used, seeded
            with seed. Default is None.
        """
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='uniform',
            rng=rng,
            seed=seed
        )


class Normal(ContinuousPrior):
    r"""A normal continuous random variable.

    The location (``loc``) keyword specifies the mean. The scale (``scale``)
    keyword specifies the standard deviation.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    scale: array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Attributes
    ----------
    name : str
        Name of random variate.
    tex : str
        LaTeX formatted name of random variate.

    Notes
    -----
    The probability density function for ``Normal`` is:

    .. math::
        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}

    for a real number :math:`x`.

    Examples
    --------
    >>> import numpy as np
    >>> from pylfi.priors import Normal

    Initialize prior distribution for random variate :math:`\theta`:

    >>> theta_prior = Normal(loc=0, scale=1, name='theta', tex=r'$\theta$', seed=42)

    Draw from prior:

    >>> theta = theta_prior.rvs(size=10)

    Evaluate probability density function:

    >>> x = np.linspace(-2, 2, 1000)
    >>> pdf = theta_prior.pdf(x)

    Display the probability density function:

    >>> theta_prior.plot_prior(x)
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='norm',
            rng=rng,
            seed=seed
        )


class Beta(ContinuousPrior):
    r"""A beta continuous random variable.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    scale: array_like, optional
        Scale parameter (default=1)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.

    Notes
    -----
    The probability density function for ``Beta`` is:

    .. math::
        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    ``Beta`` takes ``a`` and ``b`` as shape parameters.
    """

    def __init__(
        self,
        a,
        b,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(a, b),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='beta',
            rng=rng,
            seed=seed
        )


class LogNormal:
    pass


class NegativeBinomial:
    pass


class Exponential(ContinuousPrior):
    r"""An exponential continuous random variable.

    Notes
    -----
    The probability density function for ``Exponential`` is:

    .. math::
        f(x) = \exp(-x)

    for :math:`x \ge 0`.

    The probability density is defined in the “standardized” form. To shift
    and/or scale the distribution use the ``loc`` and ``scale`` parameters.

    A common parameterization for ``Exponential`` is in terms of the
    rate parameter ``lambda``, such that ``pdf = lambda * exp(-lambda * x)``.
    This parameterization corresponds to using ``scale = 1 / lambda``.
    """

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(),
            loc=loc,
            scale=scale,
            name=name,
            tex=tex,
            distr_name='expon',
            rng=rng,
            seed=seed
        )


class Gamma:
    pass


class InvGamma:
    pass


class Dirichlet:
    pass


class Geometric:
    pass


class Multinomial:
    pass


class Randint(DiscretePrior):
    r"""A uniform discrete random variable.

    Notes
    -----
    The probability mass function for ``Randint`` is:

    .. math::
        f(k) = \frac{1}{high - low}

    for ``k = low, ..., high - 1``.

    ``Randint`` takes ``low`` and ``high`` as shape parameters.

    The probability mass function is defined in the “standardized” form.
    To shift distribution use the ``loc`` parameter.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.
    """

    def __init__(
        self,
        low,
        high,
        loc=0.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(low, high),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='randint',
            rng=rng,
            seed=seed
        )


class Binomial(DiscretePrior):
    r"""A binomial discrete random variable.

    Notes
    -----
    The probability mass function for ``Binomial`` is:

    .. math::
       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for ``k`` in ``{0, 1,..., n}``, :math:`0 \leq p \leq 1`

    ``Binomial`` takes ``n`` and ``p`` as shape parameters, where
    ``p`` is the probability of a single success and ``1 - p`` is
    the probability of a single failure.

    The probability mass function is defined in the “standardized” form.
    To shift distribution use the ``loc`` parameter.

    Parameters
    ----------
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.
    """

    def __init__(
        self,
        n,
        p,
        loc=0.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(n, p),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='binom',
            rng=rng,
            seed=seed
        )


class Poisson(DiscretePrior):
    r"""A Poisson discrete random variable.

    Notes
    -----
    The probability mass function for ``Poisson`` is:

    .. math::
        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    ``Poisson`` takes :math:`\mu` as shape parameter. When ``mu = 0`` then at
    quantile ``k = 0``, ``pmf`` method returns ``1.0``.

    The probability mass function is defined in the “standardized” form.
    To shift distribution use the ``loc`` parameter.

    Parameters
    ----------
    mu : float
        Shape parameter
    loc : array_like, optional
        Location parameter (default=0)
    name : str
        Name of random variate. Default is None (which will raise an ``Error``).
    tex : raw str literal, optional
        LaTeX formatted name of random variate given as ``r"foo"``. Default is
        None.
    rng : Random number generator, optional
        Defines the random number generator to be used. Default is
        np.random.RandomState
    seed : {None, int}, optional
        This parameter defines the object to use for drawing random
        variates. If seed is None the RandomState singleton is used.
        If seed is an int, a new RandomState instance is used, seeded
        with seed. Default is None.
    """

    def __init__(
        self,
        mu,
        loc=0.0,
        name=None,
        tex=None,
        rng=np.random.RandomState,
        seed=None
    ):
        super().__init__(
            shape=(mu),
            loc=loc,
            name=name,
            tex=tex,
            distr_name='poisson',
            rng=rng,
            seed=seed
        )


if __name__ == "__main__":

    rv = Exponential(name='gbar_K', tex=r'$\bar{g}_K$', seed=42)
    print(rv.rvs(10))
    x = np.linspace(-2, 2, 1000)
    # rv.plot_prior(x)

    x = np.linspace(-2, 2, 1000)

    rv = Normal(loc=0, scale=0.5, name='rv', seed=42)
    print(rv.rvs(size=2))
    # rv.plot_prior(x)

    #

    rv = Uniform(loc=0, scale=1, name='rv', tex=r'rv', seed=42)
    print(rv.rvs())
    x = np.linspace(-1, 2, 1000)
    rv.plot_prior(x)

    a = b = 1
    rv = Beta(a, b, name='rv')
    print(rv.rvs())
    # rv.plot_prior(x)

    n, p = 10, 0.5
    x = np.arange(0, n + 1)
    rv = Binomial(n, p, name='rv')
    print(rv.rvs(10))
    # rv.plot_prior(x)
