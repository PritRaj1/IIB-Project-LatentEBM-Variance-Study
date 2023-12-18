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
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    return loss.item()