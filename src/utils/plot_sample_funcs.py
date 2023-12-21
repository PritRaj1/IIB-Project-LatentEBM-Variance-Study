import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

def generate_sample(GENnet, EBMnet):
    z = GENnet.sampler.sample_p0()
    z.view(z.size(0), -1, 1, 1)
    z_prior = GENnet.sampler.get_sample(z, None, EBMnet, None)
    with torch.no_grad():
        x_pred = GENnet(z_prior) 
    
    return x_pred

def save_one_sample(final_data, hyperparams, file='Vanilla Pang'):
    """
    Function to save the last generated sample from the sample grid as a PNG image.
    """
    # Save the last generated sample as a PNG image
    last_sample = final_data[-1].cpu().detach().numpy()
    plt.figure(figsize=(2, 2))
    plt.imshow(last_sample[0], cmap='gray')
    plt.axis('off')

    # Add title with hyperparameters
    title = f"EPOCHS={hyperparams[0]}, p0_SIGMA={hyperparams[1]}, GEN_SIGMA={hyperparams[2]}"
    plt.title(title, fontsize=10)

    plt.savefig(f'img/{file}/Final Sample.png')

def save_grid(final_data, hyperparams, file='Vanilla Pang', num_images=-1, name='Final Sample Grid'):
    """
    Function to save the final grid of samples from training.
    """
    img_grid = torchvision.utils.make_grid(final_data[:num_images], normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    plt.axis('off')

    # Add title with hyperparameters
    title = f"EPOCHS={hyperparams[0]}, p0_SIGMA={hyperparams[1]}, GEN_SIGMA={hyperparams[2]}"
    plt.title(title, fontsize=10)

    plt.savefig(f'img/{file}/{name}.png')

