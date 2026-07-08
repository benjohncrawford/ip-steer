import torch

def cg(H_mvp, b, max_iters = 100, tol = 1e-6):
    """
    Solves Hw = b where H_mvp is a function that calculates
    the matrix vector product between H and a vector v i.e. H_mvp(v) = H@v
    This can be used to solve linear systems of equations without needing
    to instansiate the matrix H at any point. 
    Implementation based on: https://ddrusvyat.github.io/DSC-243/part1.html#sec-4 
    """
    # value we are solving for init at zero
    w = torch.zeros_like(b)
    
    # initial residual
    r = b - H_mvp(w) 
    p = r.clone()
    
    # squared norm of the residual 
    res_mag_old = torch.sum(r * r, dim=-1, keepdim=True)

    k = 0
    while res_mag_old.max() > tol and k < max_iters:
        Hp = H_mvp(p)
        
        # Calculate step size
        eta = res_mag_old / (torch.sum(p * Hp, dim=-1, keepdim=True) + 1e-10)
        
        # Update value based on step in search direction
        w = w + eta*p
        
        # Update residual
        r = r - eta*Hp
        res_mag_new = torch.sum(r * r, dim=-1, keepdim=True)
        
        # Update search direction
        beta = res_mag_new / (res_mag_old + 1e-10)
        p = r + beta*p
        
        k+=1
        res_mag_old = res_mag_new
        
    return w

def inverse_hvp(func, w, v):
    """
    Takes in a twice continuously differentiable function and calculates the product 
    between the inverse hessian of that function at the point w and the vector v.
    """
    # A wrapper for the HVP that only takes the vector 'y'
    def hvp_wrapper(y):
        # torch.autograd.functional.hvp returns a tuple: (loss, hvp)
        # _, hvp_out = torch.autograd.functional.hvp(func, w, v=y)
        # torch.autograd.functional.vhp is significantly faster and is equivalent
        # if the function is continuously twice differential which it is in our case. 
        _, hvp_out = torch.autograd.functional.vhp(func, w, v=y)
        return hvp_out + 1e-4 * y
    
    # Solve H * y = v -> y = H^-1 * v
    return cg(hvp_wrapper, v)
        
        