import torch

class LBFGS():
    """
    Batched implementation of L-BFGS to handle independent samples simultaneously.
    Based on: https://apxml.com/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/hands-on-lbfgs-implementation
    """
    def __init__(self, m):
        self.m = m
        self.steps = []
        self.gds = []
        self.rhos = []
        
        self.gamma = None 
        self.prev_x = None
        self.prev_grad = None
        
    @torch.no_grad()    
    def next(self, x, grad):
        batch_size = x.shape[0]
        q = grad.clone()
        alphas = []
        
        # initialize gamma if this is the first step
        if self.gamma is None:
            self.gamma = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)
            
        def bdot(v1, v2):
            """helper function for batched dot product: output shape (batch_size, 1)"""
            return torch.sum(v1 * v2, dim=-1, keepdim=True)
        
        # backward pass
        for s, y, rho in reversed(list(zip(self.steps, self.gds, self.rhos))):
            alpha = rho * bdot(s, q)
            alphas.append(alpha)
            q = q - alpha * y
            
        alphas.reverse()
        
        # initial Hessian approximation
        z = self.gamma * q 
        
        # forward pass
        for s, y, rho, alpha in zip(self.steps, self.gds, self.rhos, alphas):
            beta = rho * bdot(y, z)
            z = z + s * (alpha - beta)
            
        # update history
        if self.prev_x is not None and self.prev_grad is not None:
            s_k = x - self.prev_x
            y_k = grad - self.prev_grad
            
            sy = bdot(s_k, y_k)
            
            # handle curvature condition per-sample. 
            # if a sample fails the curvature check freeze its history update.
            valid_mask = sy > 1e-10
            if valid_mask.any():
                safe_s_k = torch.where(valid_mask, s_k, torch.zeros_like(s_k))
                safe_y_k = torch.where(valid_mask, y_k, torch.zeros_like(y_k))
                
                # Avoid division by zero for failed samples by setting rho=0
                safe_rho = torch.where(valid_mask, 1.0 / sy, torch.zeros_like(sy))
                
                self.steps.append(safe_s_k)
                self.gds.append(safe_y_k)
                self.rhos.append(safe_rho)
                
                if len(self.steps) > self.m:
                    self.steps.pop(0)
                    self.gds.pop(0)
                    self.rhos.pop(0)
                    
                yy = bdot(safe_y_k, safe_y_k)
                
                # Update gamma only for valid samples
                new_gamma = torch.where(yy > 1e-10, sy / yy, self.gamma)
                self.gamma = torch.where(valid_mask, new_gamma, self.gamma)

        self.prev_x = x.clone()
        self.prev_grad = grad.clone()
        
        return -z
    
    @torch.no_grad()
    def reset(self):
        self.steps = []
        self.gds = []
        self.rhos = []
        
        self.gamma = None 
        self.prev_x = None
        self.prev_grad = None