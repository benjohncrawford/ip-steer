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
        delta = 2,
        eps = 0.15,
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
        self.X_feas = pos_X[0]
        # self.X_feas = self.find_init_feas(target_device = pos_X.device)
        return self
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
    
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0. or self.check_feasible(X).all(): 
            return X
        return self.solve(X)
    
    def obj(self, X, X_0, eta):
        diff = X-X_0
        return 0.5*eta*torch.sum(torch.square(diff), dim=-1) - torch.log(self.clf.forward(X) - (0.5 + self.eps) + 1e-8)

    def solve(self, X_0: Tensor, tol = 1e-6, max_iter = 100) -> Tensor:
        self.clf.to(X_0.device)
        error = 10e6
        X = self.X_feas.to(X_0.device).unsqueeze(0).expand_as(X_0).clone()
        eta = self.eta_0
        prev_X = X_0
        
        outer_k = 0
        with torch.enable_grad():
            inner_k = 0
            while outer_k <= 10:
                error = 10e6
                while error >= tol and inner_k <= max_iter:
                    X = X.detach().requires_grad_(True)
                    obj_wrapper = lambda x: self.obj(x, X_0, eta).sum()
                    
                    y = obj_wrapper(X)
                    y = y.sum()
                    grad_y = torch.autograd.grad(y, X, create_graph=True)[0]
                    X = X - inverse_hvp(obj_wrapper, X, grad_y)
                    error = torch.norm(prev_X.detach() - X.detach(), dim=-1).max().item()
                    prev_X = X

                    inner_k += 1
                print("-------------------------------")
                print(f"Iteration {outer_k}:\nerror: {error}\nX:{X}\neta:{eta}\nobj: {y}\nh(a): {self.clf.forward(X)}\nfeasible: {self.check_feasible(X)}")
                print("-------------------------------")
                outer_k += 1
                eta = eta * self.delta
        return X.detach()

    def check_feasible(self, X):
        self.clf.to(X.device)
        return self.clf.forward(X) >= (0.5 + self.eps)

    # def find_init_feas(self, target_device) -> Tensor:
    #     X_0 = torch.zeros(2048, requires_grad = True, device=target_device)
    #     return self.solve(X_0, eta_0 = 0, max_iter=1000)

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
    