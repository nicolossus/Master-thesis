class _KnuthF:
    r"""Class which implements the function minimized by knuth_bin_width

    Parameters
    ----------
    data : array_like, one dimension
        data to be histogrammed

    Notes
    -----
    the function F is given by

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    See Also
    --------
    knuth_bin_width
    """

    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

        # import here rather than globally: scipy is an optional dependency.
        # Note that scipy is imported in the function which calls this,
        # so there shouldn't be any issue importing here.
        from scipy import special

        # create a reference to gammaln to use in self.eval()
        self.gammaln = special.gammaln

    def bins(self, M):
        """Return the bin edges given a width dx"""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function

        Parameters
        ----------
        dx : float
            Width of bins

        Returns
        -------
        F : float
            evaluation of the negative Knuth likelihood function:
            smaller values indicate a better fit.
        """
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M) +
                 self.gammaln(0.5 * M) -
                 M * self.gammaln(0.5) -
                 self.gammaln(self.n + 0.5 * M) +
                 np.sum(self.gammaln(nk + 0.5)))


def _freedman_diaconis_rule(data):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.

    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    https://stats.stackexchange.com/questions/798/
    https://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy

    The Freedman-Diaconis rule can be used to select the width of the bins
    to be used in a histogram.

    The general equation for the rule is
            Bin width = 2 * IQR(x) * n^(-1/3),
    where IQR(x) is the interquartile range of the data and n is the number
    of observations in the sample x.

    The number of bins is then
            Number of bins = (max - min) / Bin width,
    where max is the maximum and min is the minimum value of the data.
    """
    pass
