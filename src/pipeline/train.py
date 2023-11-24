import torch

def train_generator(x, zK_GEN, GENnet, GENoptimiser, lossG):
    """
    Function to train the generator.
    """
    GENoptimiser.zero_grad()
    x_pred = GENnet(zK_GEN.detach()) + (GENnet.lkhood_sigma * torch.randn_like(x))
    loss_gen = lossG(x, x_pred, GENnet.lkhood_sigma) 
    loss_gen.backward()
    GENoptimiser.step()

    return loss_gen.item()

def train_EBM(zK_EBM, zK_GEN, EBMnet, EBMoptimiser, lossE):
    """
    Function to train the EBM.
    """
    EBMoptimiser.zero_grad()
    loss_ebm = lossE(zK_EBM, zK_GEN, EBMnet)
    loss_ebm.backward()
    EBMoptimiser.step()
    
    return loss_ebm.item()


def train_step(x, GENnet, EBMnet, GENoptimiser, EBMoptimiser, Sampler, lossG, lossE):      
    """
    Function to perform one training step for the generator and the EBM.
    """ 
    # 1a. Sample from latent prior p0(z)
    z0 = Sampler.sample_p0()
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_GEN = Sampler.get_sample(z0, x, GENnet, EBMnet)
    
    ###
    # 2. Train generator
    loss_GEN = train_generator(x, zK_GEN, GENnet, GENoptimiser, lossG)
    
    # 3. Train EBM
    loss_EBM = train_EBM(zK_EBM, zK_GEN, EBMnet, EBMoptimiser, lossE)

    return loss_EBM, loss_GEN

