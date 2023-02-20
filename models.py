from __future__ import annotations
# don't think it is needed unless to change initialized values
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.nn import ModuleList
import gpytorch
from gpytorch import settings
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import exact_prediction_strategies
from gpytorch.module import Module
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode

## RBFKernel modifications:
def postprocess_rbf(dist_mat):   # might need to epsilon it
    return dist_mat.div(-2).exp()

class RBFKernel_modified(Kernel):
    has_lengthscale = True
    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = torch.mul(x1,torch.pow(self.lengthscale,0.5))
            x2_ = torch.mul(x2,torch.pow(self.lengthscale,0.5))
            return postprocess_rbf(self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params))
            
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params),
        )



class ExactGPModel_standard(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super(ExactGPModel_standard, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # can fix mean and constraints priors
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel_standardScaleK(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super(ExactGPModel_standardScaleK, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # can fix mean and constraints priors
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel_modLengthscale(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super(ExactGPModel_modLengthscale, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = RBFKernel_modified(ard_num_dims) # scaleKernel -> decorates kernel with a scale (ways of ajusting posterior volatility) ; can fix priors on kernel parameters

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel_modLengthscaleScaleK(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        super(ExactGPModel_modLengthscaleScaleK, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel_modified(ard_num_dims)) # scaleKernel -> decorates kernel with a scale (ways of ajusting posterior volatility) ; can fix priors on kernel parameters

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)