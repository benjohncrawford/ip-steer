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
from ..utils.ihvp import inverse_hvp


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
        self.X_feas = None
        
        # Initial eta for the interior point method should be very small 
        self.eta_0 = eta_0

        # Factor by which to multiply eta at each step MUST BE GREATER THAN ONE
        self.delta = delta

        # Desired distance/probability into the safety region 
        self.eps = eps
                
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'BaseIPSteer':
        self.clf.fit(pos_X, neg_X_or_labels)
        self.X_feas = torch.zeros(2048, requires_grad = True, device = pos_X.device)
        self.X_feas = self.find_init_feas(target_device = pos_X.device)
        return self
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
    
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0.: 
            return X
        return self.solve(X, self.eta_0)
    
    def obj(self, X, X_0, eta):
        diff = X-X_0
        return eta*torch.sum(torch.square(diff)) + torch.log(self.clf.forward(X))

    def solve(self, X_0: Tensor, eta_0 = 1e-6, tol = 1e-6, max_iter = 10) -> Tensor:
        self.clf.to(X_0.device)
        error = 10e6
        X = self.X_feas.to(X_0.device).clone()
        eta = eta_0
        prev_X = X_0
        
        k = 0
        with torch.enable_grad():
            while error >= tol and k <= max_iter:
                X = X.detach().requires_grad_(True)
                obj_wrapper = lambda x: self.obj(x, X_0, eta)
                
                y = obj_wrapper(X)
                grad_y = torch.autograd.grad(y, X, create_graph=True)[0]
                X = X - inverse_hvp(obj_wrapper, X, grad_y)
                error = torch.norm(prev_X.detach() - X.detach())
                prev_X = X
                eta = eta * self.delta
                k += 1
        return X.detach()

    def find_init_feas(self, target_device) -> Tensor:
        X_0 = torch.zeros(2048, requires_grad = True, device=target_device)
        return self.solve(X_0, 0)

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
    