import torch
import os

def sample_zK(x, GENnet, EBMnet):
    """
    Function to sample from the parametised posterior and prior distributions.
    """
    # 1a. Sample from latent prior p0(z)
    z0 = GENnet.sampler.sample_p0()

    # 1b. Sample from posterior p_theta(z | x) and prior p_a(z)
    zK_EBM = GENnet.sampler.get_sample(z0, None, EBMnet, None).detach()
    zK_GEN = GENnet.sampler.get_sample(z0, x, GENnet, EBMnet).detach()
    
    return zK_EBM, zK_GEN

def update_parameters(loss, optimiser):
    """
    Function to update the parameters of the model.
    """
    # Save the gradients of the loss for each sample in the batch
    loss_gradients = torch.zeros_like(loss)
    for batch_idx, l in enumerate(loss):
        grad_l = torch.autograd.grad(l.mean(), optimiser.param_groups[0]['params'], retain_graph=True)[0]
        loss_gradients[batch_idx] = torch.norm(grad_l)
    
    # Update the parameters
    optimiser.zero_grad()
    loss = loss.mean()
    loss.backward()
    optimiser.step()
    
    # Loss gradients is a tensor of size batch_size, contatining the gradients of the loss for each sample in the batch
    return loss.item(), loss_gradients

def create_results_directory(file, subfile):
    """
    Function to create the results directory if it doesn't exist.
    """
    # Make sure files exist, if not create them
    if not os.path.exists(f'results/{file}/'):
        os.makedirs(f'results/{file}/')
    if not os.path.exists(f'results/{file}/{subfile}/'):
        os.makedirs(f'results/{file}/{subfile}/')

    if not os.path.exists(f'img/{file}/'):
        os.makedirs(f'img/{file}/')
    if not os.path.exists(f'img/{file}/{subfile}/'):
        os.makedirs(f'img/{file}/{subfile}/')