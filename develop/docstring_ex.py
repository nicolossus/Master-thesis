from typing import Any, Dict, List, Optional


def sample(
    model: Model,
    sampler_type: Optional[str] = None,
    num_samples: int = 1000,
    num_chains: int = 10,
    burn_in: int = 100,
    observed: Optional[Dict[str, Any]] = None,
    state: Optional[flow.SamplingState] = None,
    xla: bool = False,
    use_auto_batching: bool = True,
    sampler_methods: Optional[List] = None,
    trace_discrete: Optional[List[str]] = None,
    seed: Optional[int] = None,
    include_log_likelihood: bool = False,
    **kwargs,
):
    """
    Perform MCMC sampling using NUTS (for now).
    Parameters
    ----------
    model : pymc4.Model
        Model to sample posterior for
    sampler_type : Optional[str]
        The step method type for the model
    num_samples : int
        Num samples in a chain
    num_chains : int
        Num chains to run
    burn_in : int
        Length of burn-in period
    observed : Optional[Dict[str, Any]]
        New observed values (optional)
    state : Optional[pymc4.flow.SamplingState]
        Alternative way to pass specify initial values and observed values
    xla : bool
        Enable experimental XLA
    **kwargs: Dict[str, Any]
        All kwargs for kernel, adaptive_step_kernel, chain_sample method
    use_auto_batching : bool
        WARNING: This is an advanced user feature. If you are not sure how to use this, please use
        the default ``True`` value.
        If ``True``, the model's total ``log_prob`` will be automatically vectorized to work across
        multiple indepedent chains using ``tf.vectorized_map``. If ``False``, the model is assumed
        be defined in vectorized way. This means that every distribution has the proper
        ``batch_shape`` and ``event_shape``s so that all the outputs from each distribution's
        ``log_prob`` will broadcast with each other, and that the forward passes through the model
        (prior and posterior predictive sampling) all work on values with any value of
        ``batch_shape``. Achieving this is a hard task, but it enables the model to be safely
        evaluated in parallel across all chains in MCMC, so sampling will be faster than in the
        automatically batched scenario.
    trace_discrete : Optional[List[str]]
        INFO: This is an advanced user feature.
        The pyhton list of variables that should be casted to tf.int32 after sampling is completed
    seed : Optional[int]
        A seed for reproducible sampling
    include_log_likelihood : bool, default=False
       Include log likelihood in trace
    Returns
    -------
    Trace : InferenceDataType
        An ArviZ's InferenceData object with the groups: posterior, sample_stats and observed_data
    Examples
    --------
    Let's start with a simple model. We'll need some imports to experiment with it.
    >>> import pymc4 as pm
    >>> import numpy as np
    This particular model has a latent variable `sd`
    >>> @pm.model
    ... def nested_model(cond):
    ...     sd = yield pm.HalfNormal("sd", 1.)
    ...     norm = yield pm.Normal("n", cond, sd, observed=np.random.randn(10))
    ...     return norm
    Now, we may want to perform sampling from this model. We already observed some variables and we
    now need to fix the condition.
    >>> conditioned = nested_model(cond=2.)
    Passing ``cond=2.`` we condition our model for future evaluation. Now we go to sampling.
    Nothing special is required but passing the model to ``pm.sample``, the rest configuration is
    held by PyMC4.
    >>> trace = sample(conditioned)
    Notes
    -----
    Things that are considered to be under discussion are overriding observed variables. The API
    for that may look like
    >>> new_observed = {"nested_model/n": np.random.randn(10) + 1}
    >>> trace = sample(conditioned, observed=new_observed)
    This will give a trace with new observed variables. This way is considered to be explicit.
    """
    # assign sampler is no sampler_type is passed``
    sampler_assigned: str = auto_assign_sampler(model, sampler_type)

    try:
        sampler = reg_samplers[sampler_assigned]
    except KeyError:
        _log.warning(
            "The given sampler doesn't exist. Please choose samplers from: {}".format(
                list(reg_samplers.keys())
            )
        )
        raise

    # TODO: keep num_adaptation_steps for nuts/hmc with
    # adaptive step but later should be removed because of ambiguity
    if any(x in sampler_assigned for x in ["nuts", "hmc"]):
        kwargs["num_adaptation_steps"] = burn_in

    sampler = sampler(model, **kwargs)

    # If some distributions in the model have non default proposal
    # generation functions then we lanuch compound step instead of rwm
    if sampler_assigned == "rwm":
        compound_required = check_proposal_functions(
            model, state=state, observed=observed)
        if compound_required:
            sampler_assigned = "compound"
            sampler = reg_samplers[sampler_assigned](model, **kwargs)

    if sampler_assigned == "compound":
        sampler._assign_default_methods(
            sampler_methods=sampler_methods, state=state, observed=observed
        )

    return sampler(
        num_samples=num_samples,
        num_chains=num_chains,
        burn_in=burn_in,
        observed=observed,
        state=state,
        use_auto_batching=use_auto_batching,
        xla=xla,
        seed=seed,
        trace_discrete=trace_discrete,
        include_log_likelihood=include_log_likelihood,
    )
