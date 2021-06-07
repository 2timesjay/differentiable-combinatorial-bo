from typing import List, Optional
import functools

import torch
from torch.nn import Module
from torch import Tensor
    
from botorch.models.transforms.input import InputTransform

from fast_soft_sort import pytorch_ops
import perturbations

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
dtype = torch.double

def ranks(inputs, dim=-1):
    """Returns the ranks of the input values among the given axis."""
    return inputs.argsort(dim).argsort(dim).type(inputs.dtype)

def approximate_round(X: Tensor, tau: float = 1e-3) -> Tensor:
    r"""Diffentiable approximate rounding function.
    This method is a piecewise approximation of a rounding function where
    each piece is a hyperbolic tangent function.
    Args:
        X: The tensor to round to the nearest integer (element-wise).
        tau: A temperature hyperparameter.
    Returns:
        The approximately rounded input tensor.
    """
    offset = X.floor()
    scaled_remainder = (X - offset - 0.5) / tau
    rounding_component = (torch.tanh(scaled_remainder) + 1) / 2
    return offset + rounding_component

def soft_round(X: Tensor, tau: float = 1e-3) -> Tensor:
    """Similar to the all-pairs comparison approach used by:
    
    Qin, T., Liu, T.-Y., and Li, H. A general approximation
    framework for direct optimization of information retrieval
    measures. Information retrieval, 13(4):375–397, 2010.
    """
    n = X.shape[0]
    d = X.shape[1]
    differences = X.reshape(n, 1, d) - X.reshape(n, d, 1)
    rounded_differences = approximate_round((differences + 1) / 2, tau=tau)
    return (rounded_differences.sum(dim=1) - 0.5)

def hard_round(X: Tensor) -> Tensor:
    return ranks(X)

class SoftRoundLayer(torch.nn.Module):
    def __init__(self, tau: float):
        self.tau = tau
        super(SoftRoundLayer, self).__init__()

    def forward(self, x):
        return soft_round(x, tau=self.tau)
    
class HardRoundLayer(torch.nn.Module):
    def __init__(self):
        super(HardRoundLayer, self).__init__()

    def forward(self, x):
        return hard_round(x)
    
class RankRoundTransform(InputTransform, Module):
    """Round points in [0,1]^d to permutations (vertices of the permutahedron).
    
    Rounding can be hard and non-differentiable, or approximate and differentiable
    depending on the torch layer used.
    """

    def __init__(
        self,
        indices: List[int],
        dim: Optional[int] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_preprocess: bool = False,
        hard: bool = False,
        tau: float = 1e-3,
    ) -> None:
        r"""Initialize transform.
        Args:
            indices: The indices of the constrained parameters.
            bound: The bound on the L1 norm of constrained parameters.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when setting training inputs on the mode. Default: False
        """
        super().__init__()
        self.dim = dim or len(indices)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.register_buffer("indices", torch.tensor(indices, device=device, dtype=torch.long))
        self.layer = HardRoundLayer() if hard else SoftRoundLayer(tau=tau)

    def transform(self, X: Tensor) -> Tensor:
        r"""Project the inputs.
        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.
        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_projected = X.clone()
        X_constrained = X_projected[..., self.indices].reshape(-1, len(self.indices))
        X_constrained = self.layer(X_constrained)
        if X_projected.ndim == 3:
            X_constrained = X_constrained.reshape(X_projected.shape[0], X_projected.shape[1], len(self.indices))
        X_projected[..., self.indices] = X_constrained
        return X_projected/self.dim
    
class SoftRankTransform(InputTransform, Module):
    """Approximately round points in [0,1]^d to permutations (vertices of the permutahedron).
    
    Uses Fast Differentiable Sorting and Ranking from
    BlondelTeboulBerthetDjolonga2020"""
    def __init__(
        self,
        indices: List[int],
        regularization_strength: float,
        regularization_type: str,
        dim: Optional[int] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_preprocess: bool = False,
    ) -> None:
        r"""Initialize transform.
        Args:
            indices: The indices of the constrained parameters.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when setting training inputs on the mode. Default: False
        """
        self.dim = dim or len(indices)
        super().__init__()
        self.regularization_strength = regularization_strength 
        self.regularization_type = regularization_type
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.register_buffer("indices", torch.tensor(indices, device=device, dtype=torch.long))
        self.layer = pytorch_ops.soft_rank

    def transform(self, X: Tensor) -> Tensor:
        r"""Project the inputs.
        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.
        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_projected = X.clone()
        X_constrained = X_projected[..., self.indices].reshape(-1, len(self.indices))
        X_constrained = self.layer(
            X_constrained, 
            regularization_strength=self.regularization_strength, 
            regularization=self.regularization_type,
        )
        if X_projected.ndim == 3:
            X_constrained = X_constrained.reshape(X_projected.shape[0], X_projected.shape[1], len(self.indices))
        X_projected[..., self.indices] = X_constrained
        return X_projected/self.dim

class PerturbedRankTransform(InputTransform, Module):
    """Approximately round points in [0,1]^d to permutations (vertices of the permutahedron).
    
    Rounding can be hard and non-differentiable, or approximate and differentiable
    depending on the torch layer used.
    
    Based on Learning with Differentiable Perturbed Optimizers
    BerthetBlondelTeboulCuturiVertBach2020
    """
    def __init__(
        self,
        indices: List[int],
        num_samples: int,
        sigma: float,
        noise: str="gumbel",
        dim: Optional[int] = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_preprocess: bool = False,
    ) -> None:
        r"""Initialize transform.
        Args:
            indices: The indices of the constrained parameters.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True
            transform_on_preprocess: A boolean indicating whether to apply the
                transform when setting training inputs on the mode. Default: False
        """
        super().__init__()
        self.dim = dim or len(indices)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_preprocess = transform_on_preprocess
        self.register_buffer("indices", torch.tensor(indices, device=device, dtype=torch.long))
#         optimizer = lambda input: is_top_k(ranks(input), K)
        optimizer = ranks
        pert_ranks = perturbations.perturbed(
            optimizer,
            num_samples=num_samples,
            sigma=sigma,
            noise=noise,
            batched=True,
            device=device,
        )
        self.layer = pert_ranks

    def transform(self, X: Tensor) -> Tensor:
        r"""Project the inputs.
        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.
        Returns:
            A `batch_shape x n x d`-dim tensor of rounded inputs.
        """
        X_projected = X.clone()
        X_constrained = X_projected[..., self.indices].reshape(-1, len(self.indices))
        X_constrained = self.layer(X_constrained)
        if X_projected.ndim == 3:
            X_constrained = X_constrained.reshape(X_projected.shape[0], X_projected.shape[1], len(self.indices))
        X_projected[..., self.indices] = X_constrained
#         return X_projected/self.dim
        return torch.clamp((X_projected)/self.dim, 0.0, 1.0)