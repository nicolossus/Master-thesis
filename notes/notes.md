
**Chapter name ideas**
* Post-hoc Adjustments
  * LRA (Linear regression adjustment)


**Transformation of variables.** When implementing MCMC, it is common advice to transform problems to
have unbounded support (Hogg and Foreman-Mackey, 2018), although this has not been discussed in SBI papers
or implemented in accompanying code. We found that without this transformation, MCMC sampling could get
stuck in endless loops, e.g., on the Lotka-Volterra task. Apart from the transformation to unbounded space, we
found z-scoring of parameters and data to be crucial for some tasks. *(SBI benchmark paper)*

* check out `journal.Wass_convergence_plot()` in `abcpy`

* quantile epsilon (adaptive)
* Epsilon–acceptance rate

* HH inference: the experimental data HH used?

* summary statistic: learned vs domain knowledge


**Package additions**
* tqdm
* colorlog
* requests


sphinx req.txt

Sphinx~=3.5.1
Pallets-Sphinx-Themes~=1.2.2
sphinxcontrib-log-cabinet~=1.0.1
sphinxcontrib-napoleon
sphinx-issues~=1.2.0
packaging~=19.2
sphinx_rtd_theme==0.5.1
readthedocs-sphinx-search==0.1.0
