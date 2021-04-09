.. _gettingstarted:

2. Getting Started
==================

Here, we explain how to use ABCpy to quantify parameter uncertainty of a probabilistic model given some observed
dataset. If you are new to uncertainty quantification using Approximate Bayesian Computation (ABC), we recommend you to
start with the `Parameters as Random Variables`_ section.

Moreover, we also provide an interactive notebook on Binder guiding through the basics of ABC with ABCpy; without
installing that on your machine.
Please find it `here <https://mybinder.org/v2/gh/eth-cscs/abcpy/master?filepath=examples>`_.

Parameters as Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, if we have measurements of the height of a group of grown up humans and it is also known that a Gaussian
distribution is an appropriate probabilistic model for these kind of observations, then our observed dataset would be
measurement of heights and the probabilistic model would be Gaussian.

..
  .. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
      :language: python
      :lines: 86-98, 103-105
      :dedent: 4
