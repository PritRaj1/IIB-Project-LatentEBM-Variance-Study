import torch
import matplotlib.pyplot as plt

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
    GENoptimiser.zero_grad()
    x_pred = GENnet(zK_GEN.detach()) + (GENnet.lkhood_sigma * torch.randn_like(x))
    loss_gen = lossG(x, x_pred, GENnet.lkhood_sigma) 
    loss_gen.backward()
    GENoptimiser.step()
    
    # 3. Train EBM
    EBMoptimiser.zero_grad()
    loss_ebm = lossE(zK_EBM, zK_GEN, EBMnet)
    loss_ebm.backward()
    EBMoptimiser.step()
    
    return loss_ebm.item(), loss_gen.item()  

def generate_sample(Sampler, GENnet, EBMnet):
    z = Sampler.sample_p0()
    z_prior = Sampler.get_sample(z, None, EBMnet, None)
    with torch.no_grad():
        x_pred = GENnet(z_prior) 
    
    return x_pred

def save_final_sample(final_x, hyperparams):
    """
    Function to save the last generated sample as a PNG image.
    """
    # Save the last generated sample as a PNG image
    last_sample = final_x[-1].cpu().detach().numpy()
    plt.imshow(last_sample[0], cmap='gray')
    plt.axis('off')

    # Add title with hyperparameters
    title = f"EPOCHS={hyperparams[0]}, p0_SIGMA={hyperparams[1]}, GEN_SIGMA={hyperparams[2]}"
    plt.title(title)

    plt.savefig('img/Final Vanilla Pang Sample.png')