import torch
import torch.nn as nn

class temperedGenerator(nn.Module):
    """
    Top-down generator with power posteriors (thermodynamic integration)

    Args:
    - input_dim (int): the dimensionality of the input noise vector
    - feature_dim (int): the dimensionality of the feature maps in the generator
    - output_dim (int): the dimensionality of the output data
    - lkhood_sigma (float): the standard deviation of the likelihood distribution
    - langevin_steps (int): the number of steps the langevin sampler takes
    - langevin_s (float): the step size of the langevin sampler
    - temp_schedule (str): the name of the temperature schedule
    - num_temps (int): the number of temperatures in the temperature schedule
    
    Methods:
    - forward(z): generates a sample from the generator
    - grad_log_fn(z, x, EBM_model): computes the gradient of the log posterior: log[p(x | z)^t * p(z)] w.r.t. z
    - temperature_schedule(schedule_name, num_temps): sets the temperature schedule
    """
    def __init__(self, input_dim, feature_dim, output_dim, lkhood_sigma, langevin_steps=20, langevin_s=0.1, temp_schedule='uniform', num_temps=10):
        super().__init__()
        
        self.s = langevin_s # This is s in langevin sampler
        self.K = langevin_steps # Number of steps the langevin sampler takes
        self.lkhood_sigma = lkhood_sigma
        self.temperature_schedule(temp_schedule, num_temps)
        self.t_index=0
        
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
        g_z = self.layers(z)
        return g_z
    
    def grad_log_fn(self, z, x, EBM_model):
        
        # Compute gradient of t * log[p(x | z)] w.r.t z
        g_z = self.forward(z)
        log_llood = - self.schedule[self.t_index] * (torch.norm(x-g_z, dim=-1)**2) / (2.0 * self.lkhood_sigma * self.lkhood_sigma)
        grad_log_llhood = torch.autograd.grad(log_llood.sum(), z, create_graph=True)[0]
        
        # Compute gradient of log[p(z)] w.r.t z
        f_z = EBM_model.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        grad_log_prior = grad_f_z - (z / (EBM_model.p0_sigma * EBM_model.p0_sigma))
        
        return grad_log_llhood + grad_log_prior # This is GRAD log[ p(x | z)^t * p(z) ]

    def temperature_schedule(self, schedule_name, num_temps):

        if schedule_name == 'uniform':
            self.schedule = torch.tensor([i / num_temps for i in range(num_temps)])
            self.delta_t = torch.ones_like(self.schedule) * (1.0 / num_temps)
        else:
            raise ValueError('Unknown schedule name: {}'.format(schedule_name))
