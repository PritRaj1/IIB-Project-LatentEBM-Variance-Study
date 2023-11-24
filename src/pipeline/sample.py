import torch
import numpy as np
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

def generate_final_grid(Sampler, GENnet, EBMnet, num_samples):
    """
    Function to generate a grid of samples from the final model and plot it.
    """
    generated_data = torch.zeros((num_samples, 1, 28, 28))
    for i in range(num_samples):
        generated_data[i] = generate_sample(Sampler, GENnet, EBMnet).reshape(-1, 1, 28, 28)[-1]

    # Plot the grid of samples
    grid_size = int(np.sqrt(num_samples + 1))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(generated_data[i][0].cpu().detach().numpy(), cmap='gray')
        ax.axis('off')

    plt.savefig('img/Final Vanilla Pang Sample Grid.png')
    