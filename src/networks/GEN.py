import torch
import torch.nn as nn

from src.utils.grad_log_functions import vanillaGEN_grad_log_fn
from src.utils.helper_functions import sample_zK, update_parameters

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
    def __init__(self, input_dim, feature_dim, output_dim, sampler, lkhood_sigma, langevin_steps=20, langevin_s=0.1, device='cuda'):
        super().__init__()

        self.s = langevin_s
        self.K = langevin_steps
        self.lkhood_sigma = lkhood_sigma
        self.device = device

        # Langevin sampler
        self.sampler = sampler

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_dim*8, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(feature_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim*8, feature_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim*4, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.layers(z)
    
    def grad_log_fn(self, z, x, EBM_model):
        return vanillaGEN_grad_log_fn(self, z, x, EBM_model)

    def loss_fn(self, x, z):
        # Compute -log[p(x | z)]; maximize this
        x_pred = self.forward(z) + (self.lkhood_sigma * torch.randn_like(x))
        log_lkhood = (torch.norm(x-x_pred, dim=-1)**2) / (2.0 * self.lkhood_sigma * self.lkhood_sigma)
        
        return log_lkhood
    
    def train(self, x, EBM):
        """
        Train the generator.

        Args:
        - x (torch.Tensor): the batch of images
        - EBM (torch.nn.Module): the energy-based model

        Returns:
        - loss (list): the losses of the generator and EBM
        - variances (list): the mean and variance of the posterior sample variances
        - gradients (list): the mean and variance of the gradients of the generator and EBM
        """
        # Move data to device
        x = x.to(self.device)
        
        # 1. Sample from exp-tilted prior and posterior
        zK_EBM, zK_GEN = sample_zK(x, self, EBM) # Shapes are (batch_size, NUM_Z, 1, 1)

        # 2. Train generator
        loss_GEN = self.loss_fn(x, zK_GEN)
        
        # 3. Train EBM
        loss_EBM = EBM.loss_fn(zK_EBM, zK_GEN)

        # 4. Compute variances of zK_GEN across all samples (NUM_Z dimension)
        variances = torch.var(zK_GEN, dim=(1, 2, 3))

        # 5. Update parameters - get itemised loss, as well as gradients in the losses w.r.t params
        loss_EBM, lossEBM_grad = update_parameters(loss_EBM, EBM.optimiser)
        loss_GEN, lossGEN_grad = update_parameters(loss_GEN, self.optimiser)

        # 6. Compute mean and variance of gradients in the loss likelihood 
        total_grad_loss = lossEBM_grad + lossGEN_grad 
        mean_grad_loss = torch.mean(total_grad_loss)
        var_grad_loss = torch.var(total_grad_loss)

        # 7. Return losses, posterior sample variances, and gradients
        return [loss_GEN, loss_EBM], [[torch.mean(variances)], [torch.var(variances)]], [mean_grad_loss, var_grad_loss]