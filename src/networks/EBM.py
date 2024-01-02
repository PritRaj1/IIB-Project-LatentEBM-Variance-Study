import torch
import torch.nn as nn

from src.utils.grad_log_functions import EBM_grad_log_fn

class tiltedpriorEBM(nn.Module):
    """
    Tilted prior energy-based model.

    Args:
    - input_dim (int): the dimensionality of the input noise vector
    - feature_dim (int): the dimensionality of the feature space
    - output_dim (int): the dimensionality of the output
    - p0_sigma (float): the standard deviation of the prior distribution
    - langevin_steps (int): the number of Langevin steps to take
    - langevin_s (float): the step size of the Langevin sampler

    Methods:
    - forward(z): computes the energy of the input z
    - grad_log_fn(z, x, model): computes the gradient of the log prior: log[p_a(z)] w.r.t. z
    - loss_fn(z_prior, z_posterior): computes the loss function based on the prior and posterior energies
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
        
        return EBM_grad_log_fn(self, z)
    
    def loss_fn(self, z_prior, z_posterior):
        en_pos = self(z_posterior.detach())
        en_neg = self(z_prior.detach())
                
        return (en_pos - en_neg)