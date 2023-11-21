import torch

def generator_loss(x, g_z, lkhood_sigma):
    log_lkhood = (torch.norm(x-g_z, dim=-1)**2) / (2.0 * lkhood_sigma * lkhood_sigma)
    return log_lkhood.mean()

def generator_MSE_loss(x, g_z):
    return torch.nn.MSELoss()(x, g_z)