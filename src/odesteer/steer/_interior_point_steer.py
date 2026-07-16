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
from ..utils.kernels import PolyClassifier, MultiPolyClassifiers
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
                
    def fit(self, pos_Xs, neg_X_or_labels) -> 'BaseIPSteer':
        if torch.is_tensor(pos_Xs):
            pos_Xs = [pos_Xs]
            neg_X_or_labels = [neg_X_or_labels]
            
        self.clf.fit(pos_Xs, neg_X_or_labels)
            
        # Create a candidate starting point by averaging the first positive 
        # sample from every objective, giving it a decent head start.
        stacked_pos = torch.stack([x[0] for x in pos_Xs])
        candidate_X = stacked_pos.mean(dim=0).unsqueeze(0) 
        
        # Run Phase 1 to push the candidate strictly into the feasible region
        self.X_feas = self.find_init_feas(candidate_X).squeeze(0)
        
        return self
    
    def vector_field(self, X: Tensor) -> Tensor:
        return
    
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        feasible_mask = self.check_feasible(X)
        if T == 0. or feasible_mask.all(): 
            return X
        
        # Only infeasible points need to be steered so only run method on subset
        infeasible_mask = ~feasible_mask
        need_steering = X[infeasible_mask]        
        steered = self.solve(need_steering)
        
        # overwrite only the infeasible rows
        result = X.clone()
        result[infeasible_mask] = steered
        return result
    
    def obj(self, X, X_0, eta):
        diff = X-X_0
        res = 0.5*eta*torch.sum(torch.square(diff), dim=-1, keepdim=True)
        
        # Output shape: (batch_size, num_classifiers)
        clf_probs = self.clf.forward(X) 
        
        # Calculate the barrier and sum across all classifiers
        barrier = torch.log(clf_probs - (0.5 + self.eps) + 1e-8)
        res -= barrier.sum(dim=-1, keepdim=True) 
        return res

    def solve(self, X_0: Tensor, tol = 1e-4, max_outer_iter = 20, max_inner_iter = 20) -> Tensor:
        self.clf.to(X_0.device)
        X = self.get_warm_start(X_0).clone()
        eta = self.eta_0
        max_eta = 10e6
        outer_prev_X = X.clone()
        inner_prev_X = X.clone() 
        
        outer_k = 0
        outer_error = 10e6
        max_line_search_iters = 8
        tau = 0.5    # How much to shrink the step size on failure (e.g., cut in half)
        with torch.enable_grad():
            # Outer loop controls increasing eta
            while outer_k <= max_outer_iter and outer_error >= tol:
                inner_k = 0
                inner_error = 10e6
                # inner loop ensures we converge to the central path each time
                while inner_error >= tol and inner_k <= max_inner_iter:
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
                        # current_obj = self.obj(X, X_0, eta)
                        for i in range(max_line_search_iters):
                            X_proposed = X - alpha * step
                            
                            # Check if the objective actually decreased
                            # proposed_obj = self.obj(X_proposed, X_0, eta)
                            # descent_mask = proposed_obj < current_obj

                            # Check feasibility per-sample
                            feasible_mask = self.check_feasible(X_proposed)
                            if feasible_mask.ndim == 1:
                                feasible_mask = feasible_mask.unsqueeze(-1)
                                
                            # valid_mask = feasible_mask & descent_mask
                            if feasible_mask.all():
                                break
                            else:
                                # We hit or crossed the boundary or didn't decrease objective function.
                                # Shrink step size ONLY for failures.
                                alpha = torch.where(feasible_mask, alpha, alpha * tau)
                            # print(f"Line Search {i}")
                        
                        # Update X, keeping failed line-searches in their previous safe location
                        safe_X_proposed = torch.where(feasible_mask, X_proposed, X)
                        X.copy_(safe_X_proposed)
                    
                    # Compute max change between previous and current x to see if we have converged to central path
                    inner_error = self.calc_error(inner_prev_X, X)
                    # if inner_k % 10 == 0:
                    #     print("-------------------------------")
                    #     print(f"Inner Iteration {inner_k}:\nerror: {inner_error}\nX:{X}\ninner_prev_X:{inner_prev_X}\n obj: {y}")
                    #     print("-------------------------------")
                    inner_prev_X = X.clone()                    
                    inner_k += 1
                outer_error = self.calc_error(outer_prev_X, X)
                # with torch.no_grad():
                #     print("-------------------------------")
                #     print(f"Iteration {outer_k}:\nerror: {outer_error}\nX:{X}\neta:{eta}\nobj: {y}\nh(a): {self.clf.forward(X)}\nfeasible: {self.check_feasible(X)}\n dist: {torch.sum(torch.square(X-X_0), dim=-1, keepdim=True)}")
                #     print("-------------------------------")
                # Compute max change between previous and current x to see if we have converged to final solution
                outer_prev_X = X.clone()
                outer_k += 1
                eta = min(eta * self.delta, max_eta)
        return X.detach()

    def get_warm_start(self, X_0):
        X_f = self.X_feas.to(X_0.device).unsqueeze(0).expand_as(X_0)
        alphas = torch.linspace(0.01, 1.0, steps=20, device=X_0.device).view(-1, 1)

        warm_X = X_f.clone() # Fallback to X_f for elements that never become feasible

        # Go backwards so that smaller valid alphas overwrite larger ones
        for alpha in reversed(alphas):
            candidate = X_0 + alpha * (X_f - X_0)
            feas_mask = self.check_feasible(candidate)
            if feas_mask.ndim == 1:
                feas_mask = feas_mask.unsqueeze(-1)

            warm_X = torch.where(feas_mask, candidate, warm_X)

        return warm_X.clone().requires_grad_(True)

    def calc_error(self, prev, cur):
        with torch.no_grad():
            return torch.norm(prev.detach() - cur.detach(), dim=-1).max().item()

    def check_feasible(self, X):
        self.clf.to(X.device)    
        with torch.no_grad():
            # Returns True only if a sample is feasible across ALL classifiers
            return (self.clf.forward(X) >= (0.5 + self.eps + 1e-4)).all(dim=-1)

    def find_init_feas(self, X_0: Tensor, max_iters: int = 10000, lr: float = 0.01) -> Tensor:
        """
        Finds an initial feasible point by minimizing constraint violations using Adam.
        """
        self.clf.to(X_0.device)
        
        # Clone to avoid modifying the original and enable gradients
        X_feas = X_0.detach().clone().requires_grad_(True)
        
        # Use Adam for rapid convergence to the feasible region
        optimizer = torch.optim.Adam([X_feas], lr=lr)
        
        target_prob = 0.5 + self.eps
        # Convert the target probability to a target logit using the inverse sigmoid (logit) function
        # math: log(p / (1 - p))
        target_logit = torch.log(torch.tensor(target_prob / (1.0 - target_prob), device=X_0.device))
        
        for i in range(max_iters):
            # print("-------------------------------------------------")
            # print(f"Iteration: {i}")
            optimizer.zero_grad()
            # print(f"X_feas: {X_feas}")
            with torch.no_grad():
                probs = self.clf.forward(X_feas)
            logits = self.clf.predict_raw_prob(X_feas)
            # print(f"probs:{probs}")
            
            # Calculate violations: How far below the target probability are we?
            # torch.relu ensures we ONLY penalize classifiers where prob < target
            violations = torch.relu(target_logit - logits)
            # print(f"violations: {violations}")
            # If all violations are exactly 0, we are inside the intersection of all safe regions!
            if (violations == 0).all():
                print(f"→ Feasible point found in {i} iterations.")
                return X_feas.detach()
            
            # Loss is the sum of squared constraint violations
            loss = torch.sum(violations ** 2)
            # print(f"loss: {loss}")
            loss.backward()
            optimizer.step()
            # print("---------------------------------------------------")
            
        print("Warning: Phase I reached max_iters without finding a strictly feasible point.")
        return X_feas.detach()

    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError
        


class IPSteer(BaseIPSteer):
    '''
    Interior Point Steering with NormedPolyCntSketch classifier
    '''
    def _init_clf(self, **kwargs) -> MultiPolyClassifiers:
        return MultiPolyClassifiers(**kwargs)
    
    
# class RFFODESteer(BaseIPSteer):
#     '''
#     Ablation study for ODESteer with RFF classifier
#     '''
#     def _init_clf(self, **kwargs) -> RFFClassifier:
#         return RFFClassifier(**kwargs)
    