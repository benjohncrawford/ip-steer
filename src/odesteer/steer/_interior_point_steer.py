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
        alpha = 0.01,
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
        
        self.alpha = alpha
                
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
        return 0.5*eta*torch.sum(torch.square(diff), dim=-1, keepdim=True) - torch.log(self.clf.forward(X) - (0.5 + self.eps) + 1e-8)

    def solve(self, X_0: Tensor, tol = 1e-6, max_iter = 100) -> Tensor:
        self.clf.to(X_0.device)
        X = self.X_feas.to(X_0.device).unsqueeze(0).expand_as(X_0).clone()
        eta = self.eta_0
        inner_prev_X = X_0
        outer_prev_X = X_0
        
        outer_k = 0
        outer_error = 10e6
        max_line_search_iters = 15
        tau = 0.5    # How much to shrink the step size on failure (e.g., cut in half)
        with torch.enable_grad():
            # Outer loop controls increasing eta
            while outer_k <= max_iter and outer_error >= tol:
                inner_k = 0
                inner_error = 10e6
                # inner loop ensures we converge to the central path each time
                while inner_error >= tol and inner_k <= max_iter:
                    X = X.detach().requires_grad_(True)
                    
                    # Calc step summing so we can do batches
                    obj_wrapper = lambda x: self.obj(x, X_0, eta).sum()
                    y = obj_wrapper(X).sum()
                    grad_y = torch.autograd.grad(y, X, create_graph=False)[0]
                    step = inverse_hvp(obj_wrapper, X, grad_y)
                    
                    # Start full Newton step, batched
                    alpha = torch.ones((X.shape[0], 1), device=X.device) 
                    
                    # Reverse line search to ensure step does not take us out of feasible range
                    with torch.no_grad():
                        for _ in range(max_line_search_iters):
                            X_proposed = X - alpha * step

                            # Check feasibility per-sample
                            feasible_mask = self.check_feasible(X_proposed)
                            if feasible_mask.ndim == 1:
                                feasible_mask = feasible_mask.unsqueeze(-1)

                            if feasible_mask.all():
                                break
                            else:
                                # We hit or crossed the boundary. Shrink step size ONLY for failures.
                                alpha = torch.where(feasible_mask, alpha, alpha * tau)
                        
                        X.copy_(X_proposed)
                    
                    # Compute max change between previous and current x to see if we have converged to central path
                    inner_error = self.calc_error(inner_prev_X, X)
                    inner_prev_X = X

                    inner_k += 1
                # print("-------------------------------")
                # print(f"Iteration {outer_k}:\nerror: {error}\nX:{X}\neta:{eta}\nobj: {y}\nh(a): {self.clf.forward(X)}\nfeasible: {self.check_feasible(X)}")
                # print("-------------------------------")
                # Compute max change between previous and current x to see if we have converged to final solution
                outer_error = self.calc_error(outer_prev_X, X)
                inner_prev_X = X
                outer_k += 1
                eta = eta * self.delta
        return X.detach()

    def calc_error(self, prev, cur):
        return torch.norm(prev.detach() - cur.detach(), dim=-1).max().item()

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
    Interior Point Steering with NormedPolyCntSketch classifier
    '''
    def _init_clf(self, **kwargs) -> PolyClassifier:
        return PolyClassifier(**kwargs)
    
    
# class RFFODESteer(BaseIPSteer):
#     '''
#     Ablation study for ODESteer with RFF classifier
#     '''
#     def _init_clf(self, **kwargs) -> RFFClassifier:
#         return RFFClassifier(**kwargs)
    