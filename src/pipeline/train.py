import torch

def sample_zK(x, Sampler, GENnet, EBMnet):
    """
    Function to sample from the parametised posterior and prior distributions.
    """
    # 1a. Sample from latent prior p0(z)
    z0 = Sampler.sample_p0()

    # 1b. Sample from posterior p_theta(z | x) and prior p_a(z)
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None).detach()
    zK_GEN = Sampler.get_sample(z0, x, GENnet, EBMnet).detach()
    
    return zK_EBM, zK_GEN

def forwards_GENloss(x, zK_GEN, GENnet, GENoptimiser, lossG):
    """
    Function to compute generate an x_pred from the generator acting on z and compute the loss.
    """
    x_pred = GENnet(zK_GEN) + (GENnet.lkhood_sigma * torch.randn_like(x))
    return lossG(x, x_pred, GENnet.lkhood_sigma)
    
def update_parameters(loss, optimiser):
    """
    Function to update the parameters of the model.
    """
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    return loss.item()

def train_step(x, GENnet, EBMnet, GENoptimiser, EBMoptimiser, Sampler, lossG, lossE):      
    """
    Function to perform one training step for the generator and the EBM.
    """ 
    zK_EBM, zK_GEN = sample_zK(x, Sampler, GENnet, EBMnet)
    
    # 2. Train generator
    loss_GEN = forwards_GENloss(x, zK_GEN, GENnet, GENoptimiser, lossG)
    
    # 3. Train EBM
    loss_EBM = lossE(zK_EBM, zK_GEN, EBMnet)
    
    lossG = update_parameters(loss_GEN, GENoptimiser)
    lossE = update_parameters(loss_EBM, EBMoptimiser)

    return loss_EBM, loss_GEN

def train_temperature(x, GENnet, EBMnet, GENoptimiser, EBMoptimiser, Sampler, lossG, lossE):
    """
    Function to perform one training step for the generator and the EBM with the power posteriors implementation.
    """
    loss_gen = 0
    loss_ebm = 0
    
    lossG_prev = 0
    lossE_prev = 0
    
    for i in range(1, len(GENnet.schedule)):
        GENnet.t_index = i
        
        zK_EBM, zK_GEN = sample_zK(x, Sampler, GENnet, EBMnet)
        
        lossG_current = forwards_GENloss(x, zK_GEN, GENnet, GENoptimiser, lossG)
        lossE_current = lossE(zK_EBM, zK_GEN, EBMnet)
        
        # See "discretised thermodynamic integration" using trapezoid rule
        loss_gen += 0.5 * (lossG_current + lossG_prev) * GENnet.delta_t[i]
        loss_ebm += 0.5 * (lossE_current + lossE_prev) * GENnet.delta_t[i]
        
        lossG_prev = lossG_current
        lossE_prev = lossE_current
    
    # 3. Update parameters 
    lossG = update_parameters(loss_gen, GENoptimiser)
    lossE = update_parameters(loss_ebm, EBMoptimiser)
    
    return lossE, lossG
