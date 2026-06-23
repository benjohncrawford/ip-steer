'''
Proposed method by our paper
'''

from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor
from torchdiffeq import odeint

from ._base_steer import Steer
from ..utils.kernels import KernelClassifier, RFFClassifier
from ..utils.kernels import PolyClassifier


class BaseIPSteer(Steer):
    def __init__(
        self, 
        eta_0 = 10e-6,
        delta = 1.5,
        eps = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.clf = self._init_clf(**kwargs)

        # Initial feasible point - gets set during fit
        self.X_feas = 0

        # Initial eta for the interior point method should be very small 
        self.eta_0 = eta_0

        # Factor by which to multiply eta at each step MUST BE GREATER THAN ONE
        self.delta = delta

        # Desired distance/probability into the safety region 
        self.eps = eps
                
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'BaseIPSteer':
        self.clf.fit(pos_X, neg_X_or_labels)
        self.X_feas = self.find_init_feas()
        return self
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0.: 
            return X
        return self.solve(X, self.eta_0)
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
    
    def obj_grad(self, X: Tensor, eta, X_0: Tensor) -> Tensor:
        self.clf.to(X.device)
        return eta*(X - X_0) + (1/(self.clf.forward(X)-self.eps))*self.clf.grad(X)

    def obj_hess(self, X: Tensor, eta) -> Tensor:
        self.clf.to(X.device)
        return 2*eta + (1/(self.clf.forward(X)-self.eps)**2)*self.clf.grad(X)@self.clf.grad(X).T - (1/(self.clf.forward(X)-self.eps))*self.clf.kernel.H

    def solve(self, X_0: Tensor, eta_0) -> Tensor:
        k = 0
        error = 10e6
        error_eps = 1e-6
        max_iter = 10
        X = self.X_feas
        eta = eta_0
        prev_X = X_0
        while error >= error_eps and k <= max_iter:
            X = X - torch.inverse(self.obj_hess(X, eta))@self.obj_grad(X, eta, X_0)
            error = torch.norm(prev_X - X)
            prev_X = X
            eta = eta_0 * self.delta
            k+=1
        return X

    def find_init_feas(self) -> Tensor:
        X_0 = torch.zeros()
        return self.solve(X_0, 1e-6)

    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError
        


class IPSteer(BaseIPSteer):
    '''
    ODESteer used in the paper with NormedPolyCntSketch classifier
    '''
    def _init_clf(self, **kwargs) -> PolyClassifier:
        return PolyClassifier(**kwargs)
    
    
# class RFFODESteer(BaseIPSteer):
#     '''
#     Ablation study for ODESteer with RFF classifier
#     '''
#     def _init_clf(self, **kwargs) -> RFFClassifier:
#         return RFFClassifier(**kwargs)
    