import torch
import matplotlib.pyplot as plt

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