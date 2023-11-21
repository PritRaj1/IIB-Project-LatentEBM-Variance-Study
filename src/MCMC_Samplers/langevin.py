import torch

class langevin_sampler():
    """
    A class for sampling from a distribution using Langevin dynamics.

    Args:
    - p0_sigma (float): the standard deviation of the initial prior distribution
    - batch_size (int): the number of samples to generate at once
    - num_latent_samples (int): the number of latent variables to produce during a sampling loop
    - device (str): the device to use for computation (e.g. 'cpu' or 'cuda')

    Methods:
    - get_sample(initial_sample, data, model, EBMmodel=None): generates a sample using Langevin dynamics
    - sample_p0(): generates a sample from the initial prior distribution
    """
    def __init__(self, p0_sigma, batch_size, num_latent_samples, device):
        self.device = device
        self.p0_sigma = p0_sigma
        self.batch_size = batch_size
        self.num_z = num_latent_samples

    def get_sample(self, initial_sample, data, model, EBMmodel = None):
        """
        MCMC sampling using Langevin dynamics. Is used to generated samples from both the exp-tilter prior and the posterior.

        Args:
            initial_sample (tensor): The initial sample before the loop begins.
            data (tensor): The training data, which is solely used in the gradient computation for the posterior sampling procedure.
            model (class): The model used for the sampling procedure. EBM for exp-tilter prior, Generator for posterior.
            EBMmodel (class): Used solely for the posterior sampling procedure. The EBM model is also used in calculation of the posterior gradient.

        Returns:
            x_k (tensor): The final sample
        """
        x_k = initial_sample
        
        step = 0
        
        while step < model.K:
            # Compute gradient
            grad = model.grad_log_fn(x_k, data, EBMmodel)

            # Update sample
            x_k = x_k + (model.s * model.s * grad) + (torch.sqrt(torch.tensor(2)) * model.s * torch.randn_like(x_k, device=self.device))  

            step += 1             
        
        return x_k
    
    def sample_p0(self):
        """
        Sample form the initial prior distribution.

        Returns:
            z0: A normally distributed variable
        """
        return self.p0_sigma * torch.randn(*[self.batch_size, self.num_z, 1, 1], device=self.device, requires_grad=True)