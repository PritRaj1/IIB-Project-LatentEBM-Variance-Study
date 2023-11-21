import torch
import torch.nn as nn

class tiltedpriorEBM(nn.Module):
    """
    Tilted prior energy-based model.

    Args:
    - input_dim (int): the dimensionality of the input noise vector
    - output_dim (int): the dimensionality of the output data
    - p0_sigma (float): the standard deviation of the initial prior distribution, z0 ~ N(0, p0_sigma^2 * I)
    - langevin_steps (int): the number of steps the langevin sampler takes
    - langevin_s (float): the step size of the langevin sampler

    Methods:
    - forward(z): computes the energy of the input z
    - grad_log_fn(z, x, model): computes the gradient of the log posterior: log[p(x | z) * p(z)] w.r.t. z
    """
    def __init__(self, input_dim, feature_dim, output_dim, p0_sigma, langevin_steps=20, langevin_s=0.4):
        super().__init__()
        
        self.s = langevin_s 
        self.K = langevin_steps 
        self.p0_sigma = p0_sigma # Standard deviation of z0, the initial prior distribution
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
                    nn.Linear(input_dim, feature_dim), 
                    nn.ReLU(),
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, output_dim),
                )
            
    def forward(self, z):
        # Returns f_a(z)
        return self.layers(z.squeeze()).view(-1, self.output_dim, 1, 1)
    
    def grad_log_fn(self, z, x, model):
        
        # Compute gradient of log p_a(x) w.r.t. z
        f_z = self.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        
        return grad_f_z - (z / (self.p0_sigma * self.p0_sigma)) # This is GRAD log[p_a(x)]