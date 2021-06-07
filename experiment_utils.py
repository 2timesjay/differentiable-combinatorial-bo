from __future__ import annotations
import torch

from botorch.models import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
# %debug
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
import time

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_MC_SAMPLES = 512

# # Default kwargs
# OPTIMIZATION_KWARGS = {
#     "num_restarts": 10,
#     "raw_samples": 512,
#     "options": {"batch_limit": 5, "maxiter": 200},
# }

# Faster optimization. Optionally use SLSQP for perturbation stability.
OPTIMIZATION_KWARGS = {
    "num_restarts": 4,
    "raw_samples": 200,
    "options":{"batch_limit": 5, "maxiter": 50},
}

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
)
from botorch.generation.gen import gen_candidates_torch
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from torch import Tensor


def optimize_acqf_torch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.
    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additonal keyword arguments.
    Returns:
        A two-element tuple containing
        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed canidates `0,1,...,i-1`.
    Example:
        >>> # generate `q=2` candidates jointly using 20 random restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = optimize_acqf_torch(qEI, bounds, 2, 20, 512)
        >>> generate `q=3` candidates sequentially using 15 random restarts
        >>> # and 256 raw samples
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf_torch(
        >>>     qEI, bounds, 3, 15, 256, sequential=True
        >>> )
    """
    if sequential and q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf_torch(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                fixed_features=fixed_features,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    batch_initial_conditions = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
    )

    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batched_ics = batch_initial_conditions.split(batch_limit)
    for i, batched_ics_ in enumerate(batched_ics):
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_torch(
            initial_conditions=batched_ics_,
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("init_batch_limit", "batch_limit", "nonnegative")
            },
            verbose=options.get("torch_verbose", False),
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    return batch_candidates, batch_acq_values

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
dtype = torch.double

def initialize_ranking_model(train_x, train_obj, model_type="STGP", noise_se=0.0, state_dict=None, transform=None):
    # define models for objective and constraint
    input_transform = transform
    if model_type == "FNGP":
        train_yvar = torch.tensor(noise_se**2, device=device, dtype=dtype).expand_as(train_obj)
        model = FixedNoiseGP(train_x, train_obj, train_yvar, input_transform=input_transform).to(train_x)
    elif model_type == "STGP":
        model = SingleTaskGP(train_x, train_obj, input_transform=input_transform).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def generate_test_run(
    dim: int,
    obj_func,
    batch_size: int = 3,
    n_bo_batch: int = 20,
    n_sobol_batch: int = 5,
    n_trials: int = 16,
    noise_se: float = 0.0,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    optimization_kwargs = OPTIMIZATION_KWARGS,
    verbose = False,
):

    def generate_initial_data(n=10):
        # generate training data
        train_x = torch.rand(n, dim, device=device, dtype=dtype, )
        exact_obj = obj_func(train_x).unsqueeze(-1)  # add output dimension
        train_obj = exact_obj + noise_se * torch.randn_like(exact_obj, device=device, dtype=dtype)
        best_observed_value = obj_func(train_x).max().item()
        return train_x, train_obj, best_observed_value

    def obj_callable(Z):
        return Z[..., 0]

    generic_obj = GenericMCObjective(
        objective=obj_callable,
    )

    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

    def optimize_acqf_and_get_observation(acq_func, optimize_acqf_type=None):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        optimize_acqf_func = optimize_acqf_torch if optimize_acqf_type == "torch" else optimize_acqf
        # optimize
        candidates, _ = optimize_acqf_func(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size,
            **optimization_kwargs,
        )
        # observe new values 
        new_x = candidates.detach()
        exact_obj = obj_func(new_x).unsqueeze(-1)  # add output dimension
        new_obj = exact_obj + noise_se * torch.randn_like(exact_obj, device=device)
        return new_x, new_obj

    def update_random_observations(best_random):
        """Simulates a random policy by taking a the current list of best values observed randomly,
        drawing a new random point, observing its value, and updating the list.
        """
        rand_x = torch.rand(batch_size, dim, device=device)
        next_random_best = obj_func(rand_x).max().item()
        best_random.append(max(best_random[-1], next_random_best))       
        return best_random


    def run_benchmark(initial_data, model_initializer, label, **kwargs):
        best_observed_value_by_batch = []

        # call helper functions to generate initial training data and initialize model
        train_x, train_obj, best_observed_value = initial_data
        mll, model = model_initializer(train_x, train_obj)

        best_observed_value_by_batch.append(best_observed_value)

        # run n_bo_batch rounds of BayesOpt after the initial random batch
        for iteration in range(1, n_bo_batch + 1):    
            t_start = time.time()
            # fit the models
            fit_gpytorch_model(mll)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)

            # for best_f, we use the best observed noisy values as an approximation
            acqf = qNoisyExpectedImprovement(
                model=model, 
                X_baseline=train_x,
                sampler=qmc_sampler, 
                objective=generic_obj,
            )

            # optimize and get new observation
            t_opt_start = time.time()
            new_x, new_obj = optimize_acqf_and_get_observation(acqf, **kwargs)
            t_opt_end = time.time()

            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # update progress
            best_observed_value = obj_func(train_x).max().item()
            best_observed_value_by_batch.append(best_observed_value)

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll, model = model_initializer(
                train_x=train_x, 
                train_obj=train_obj, 
                state_dict=model.state_dict(),
            )
            t_end = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value for algo {label}: {best_observed_value:>4.2f}\n"
                    f"time total: {t_end - t_start:>4.2f}, just optimization: {t_opt_end - t_opt_start:>4.2f}, "
                )
            else:
                print(".", end="")
        return best_observed_value_by_batch


    def run_trials(n_trials, model_initializer, label, initial_data_list, **kwargs):
        values_by_batch_by_trial = []
        for trial in range(n_trials):
            print(f"\nTrial {trial+1:>2} of {n_trials} ", end="")
            values_by_batch = run_benchmark(
                initial_data=initial_data_list[trial], 
                model_initializer=model_initializer, 
                label=label,
                **kwargs
            )
            values_by_batch_by_trial.append(values_by_batch)
        return values_by_batch_by_trial


    initial_data_list = [generate_initial_data(n=n_sobol_batch*batch_size) for i in range(n_trials)]

    return initial_data_list, run_trials