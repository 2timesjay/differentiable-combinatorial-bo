from typing import List, Optional

import torch
from torch import Tensor

from botorch.test_functions.synthetic import Hartmann
from botorch.test_functions.synthetic import SyntheticTestFunction


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
dtype = torch.double

def ranks(inputs, dim=-1):
    """Returns the ranks of the input values among the given axis."""
    return inputs.argsort(dim).argsort(dim).type(inputs.dtype)

def corrcoef(x, y):
    vx = x - torch.mean(x, dim=-1).reshape(-1, 1)
    vy = y - torch.mean(y, dim=-1).reshape(-1, 1)
    return torch.sum(vx*vy,dim=-1)/torch.sqrt( torch.sum(vx*vx,dim=-1)* torch.sum(vy*vy,dim=-1))
    
def is_top_k(X, dim, k):
    return torch.where(
        X + .5 > (dim - k), 
        torch.tensor(1.0, device=device, dtype=dtype), 
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    
def top_k_intersection(X, targets, k):
    top_k_x = is_top_k(X, len(targets), k)
    return torch.sum(top_k_x*targets, dim=-1)

class PermutationCorrelation(SyntheticTestFunction):
    r"""Test function where some inputs are permutations.
    
    A simplified representation of users looking through a list of options
    until satisfied with diminishing utility by position.
    """

    _optimal_value = 1.0
    _check_grad_at_opt: bool = False

    def __init__(
        self, dim, n, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self.target_perms = [torch.randperm(self.dim, device=device, dtype=dtype) for _ in range(n)]
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self._optimizers = [tuple(range(0, self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # d-dim permutation. Simply pearson correlation vs. true permutation
        evaluation_transform = ranks
        return torch.max(
            torch.stack(
                [corrcoef(target_perm, evaluation_transform(X))
                 for target_perm in self.target_perms]
            ), dim=0
        ).values
    
    
class TopK(SyntheticTestFunction):
    r"""Test function where some inputs are permutations.
    
    A simplified representation of users looking through a list of options
    until satisfied with diminishing utility by position.
    """

    _optimal_value = 1.0
    _check_grad_at_opt: bool = False

    def __init__(
        self, dim, k, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self.k = k
        self.target_vals = torch.rand(self.dim, device=device, dtype=dtype)
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self._optimizers = [tuple(range(0, self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # size of intersection of top k in X with indices of top-k in target_perm.
        evaluation_transform = ranks
        opt_val = top_k_intersection(evaluation_transform(self.target_vals), self.target_vals, self.k)
        min_val = top_k_intersection(evaluation_transform(-1*self.target_vals), self.target_vals, self.k)
        return (
            top_k_intersection(evaluation_transform(X), self.target_vals, self.k) - min_val
        ) / (
            opt_val - min_val
        )
            
class TopK(SyntheticTestFunction):
    r"""Test function where some inputs are permutations.
    
    A simplified representation of users looking through a list of options
    until satisfied with diminishing utility by position.
    """

    _optimal_value = 1.0
    _check_grad_at_opt: bool = False

    def __init__(
        self, dim, k, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self.k = k
        self.target_vals = torch.rand(self.dim, device=device, dtype=dtype)
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self._optimizers = [tuple(range(0, self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # size of intersection of top k in X with indices of top-k in target_perm.
        evaluation_transform = ranks
        opt_val = top_k_intersection(evaluation_transform(self.target_vals), self.target_vals, self.k)
        min_val = top_k_intersection(evaluation_transform(-1*self.target_vals), self.target_vals, self.k)
        return (
            top_k_intersection(evaluation_transform(X), self.target_vals, self.k) - min_val
        ) / (
            opt_val - min_val
        )    

class HartmannPermutationConstrained(Hartmann):
    # Normalized with division by 3.0090, opt for dim = 8 - found by brute force.
    # Higher if higher values are used.
    _optimal_value = 1.0 
    
    def __init__(
        self, dim: int = 6, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim,
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # size of intersection of top k in X with indices of top-k in target_perm.
        return super().evaluate_true((ranks(X)/self.dim)[..., :6])/3.0090
  
# Branching Version
# def jennaton_evaluation(X: Tensor) -> Tensor:
#     return torch.where(
#         X[...,0] == 0.0,
#         torch.where(
#             X[...,1] == 0.0,
#             X[...,3]*X[...,3] + 0.1 + X[...,7],
#             X[...,4]*X[...,4] + 0.2 + X[...,7],
#         ),
#         torch.where(
#             X[...,2] == 0.0,
#             X[...,5]*X[...,5] + 0.3 + X[...,8],
#             X[...,6]*X[...,6] + 0.4 + X[...,8],
#         )
#     )

# Rank = 0 version
def jennaton_evaluation(X: Tensor) -> Tensor:
    return torch.where(
        X[...,0] == 0.0,
        (X[...,4]-0.5)*(X[...,4]-0.5) + 0.1 + X[...,8],
        torch.where(
            X[...,1] == 0.0,
            (X[...,5]-0.5)*(X[...,5]-0.5) + 0.2 + X[...,8],
            torch.where(
                X[...,2] == 0.0,
                (X[...,6]-0.5)*(X[...,6]-0.5) + 0.3 + X[...,9],
                (X[...,7]-0.5)*(X[...,7]-0.5) + 0.4 + X[...,9],
            ),
        ),
    )

class Jennaton(SyntheticTestFunction):
    _optimal_value = 0.1
    _check_grad_at_opt: bool = False

    def __init__(
        self, dim: int, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self._optimizers = [tuple(range(0, self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
        self.rank_one_hot = RankRoundTransform(indices=range(4), hard=True)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # size of intersection of top k in X with indices of top-k in target_perm.
        opt_val = 0.1
        worst_val = 2.4
        return jennaton_evaluation(self.rank_one_hot(X))
        
