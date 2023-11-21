import torch
import torch.nn as nn

class topdownGenerator(nn.Module):
    """
    Top-down generator.

    Args:
    - input_dim (int): the dimensionality of the input noise vector
    - output_dim (int): the dimensionality of the output data
    - lkhood_sigma (float): the standard deviation of the likelihood distribution
    - langevin_steps (int): the number of steps the langevin sampler takes
    - langevin_s (float): the step size of the langevin sampler

    Methods:
    - forward(z): generates a sample from the generator
    - grad_log_fn(z, x, EBM_model): computes the gradient of the log posterior: log[p(x | z) * p(z)] w.r.t. z
    """
    def __init__(self, input_dim, output_dim, lkhood_sigma, langevin_steps=20, langevin_s=0.1):
        super().__init__()
        
        self.s = langevin_s 
        self.K = langevin_steps 
        self.lkhood_sigma = lkhood_sigma
        self.grad_log_prior = 0
        
        self.layers = nn.Sequential(
                    nn.Linear(input_dim, 256), 
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim),
                )
    
    def forward(self, z):
        g_z = self.layers(z)
        return g_z
    
    def grad_log_fn(self, z, x, EBM_model):
        
        # Compute gradient of log[p(x | z)] w.r.t z
        g_z = self.forward(z)
        log_gz = -(torch.norm(x-g_z, dim=-1)**2) / (2.0 * self.lkhood_sigma * self.lkhood_sigma)
        grad_log_gz = torch.autograd.grad(log_gz.sum(), z, create_graph=True)[0]
        
        # Compute gradient of log[p(z)] w.r.t z
        f_z = EBM_model.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        grad_log_prior = grad_f_z - (z / (EBM_model.p0_sigma * EBM_model.p0_sigma))
        
        return grad_log_gz + grad_log_prior # This is GRAD log[ p(x | z) * p(z) ]