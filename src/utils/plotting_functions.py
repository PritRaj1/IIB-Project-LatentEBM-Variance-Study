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
    """
    Function to generate a batch of samples from the model.
    """
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

def save_grid(data, hyperparams, epoch, file='Vanilla Pang', subfile=None, num_images=-1, name='Final Sample Grid'):
    """
    Function to save a grid of samples from training.
    """
    plt.figure(figsize=(10, 10))
    img_grid = torchvision.utils.make_grid(data[:num_images], normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    plt.axis('off')

    # Add title with hyperparameters
    title = f"EPOCHS={epoch}/{hyperparams[0]}, p0_SIGMA={hyperparams[1]}, GEN_SIGMA={hyperparams[2]}"
    plt.title(title, fontsize=10)


    if subfile is None:
        plt.savefig(f'img/{file}/{name}.png')
    else:
        plt.savefig(f'img/{file}/{subfile}/{name}.png')

def plot_posterior_metrics(avg_var_posterior, var_var_posterior, temperatures, cmap, FILE):
    """
    Function to plot the expected variance and variance of variances of zK_GEN for each temperature.
    """
    # Initialise plots for expected variance and variance of variances in ZK_GEN, the posterior samples
    avg_fig, avg_axs = plt.subplots(1, 1, figsize=(18, 6))
    avg_fig.suptitle("Expected Variance of Generated Samples")
    avg_axs.set_xlabel("Epoch")
    avg_axs.set_ylabel("Expected Variance")

    var_fig, var_axs = plt.subplots(1, 1, figsize=(18, 6))
    var_fig.suptitle(r"Variance of Variance of Generated Samples, $\mathrm{Var}_{p(\mathbf{x})}\left[\mathrm{Var}_{p(\mathbf{z}|\mathbf{x})}\left[ \mathbf{z} \right] \right]$")
    var_axs.set_xlabel("Epoch")
    var_axs.set_ylabel(r"$\mathrm{Var}_{p(\mathbf{x})}\left[\mathrm{Var}_{p(\mathbf{z}|\mathbf{x})}\left[ \mathbf{z} \right] \right]$")

    # Plot the expected variance and variance of variances of zK_GEN for each temperature
    for idx, temp in enumerate(temperatures):
        avg_axs.plot(avg_var_posterior[:, idx].cpu().detach().numpy(), label=f"Temperature = {temp}", color=cmap(idx / len(temperatures)))
        var_axs.plot(var_var_posterior[:, idx].cpu().detach().numpy(), label=f"Temperature = {temp}", color=cmap(idx / len(temperatures)))

    avg_axs.legend()
    avg_fig.savefig(f'img/{FILE}/Expected Variance.png')
    var_axs.legend()
    var_fig.savefig(f'img/{FILE}/Variance of Variance.png')

def plot_gradLoss_metrics(expected_gradLoss, variance_gradLoss, FILE):
    """
    Function to plot the expected variance and variance of variances of the marginal log-likelihood.
    """
    # Initialise plots for expected variance and variance of variances in grad_loss
    avg_fig, avg_axs = plt.subplots(1, 1, figsize=(18, 6))
    avg_fig.suptitle(r"Expected Variance of Gradient in Loss" + "\n" + 
                    r"$\mathrm{E}\left[\mathrm{Var}\left[ \nabla _\theta \log(p(\mathbf{x}|\theta)) \right] \right]$")
    avg_axs.set_xlabel("Epoch")
    avg_axs.set_ylabel(r"$\mathrm{E}\left[\mathrm{Var}\left[ \nabla _\theta \log(p(\mathbf{x}|\theta)) \right] \right]$")

    var_fig, var_axs = plt.subplots(1, 1, figsize=(18, 6))
    var_fig.suptitle("Variance of Variance of Gradient in Loss" + "\n" +
                    r"$\mathrm{Var}\left[\mathrm{Var}\left[ \nabla _\theta \log(p(\mathbf{x}|\theta)) \right] \right]$")
    var_axs.set_xlabel("Epoch")
    var_axs.set_ylabel(r"$\mathrm{Var}\left[\mathrm{Var}\left[ \nabla _\theta \log(p(\mathbf{x}|\theta)) \right] \right]$")
    
    # Plot the expected variance and variance of variances of zK_GEN for each temperature
    avg_axs.loglog(expected_gradLoss.cpu().detach().numpy(), color='red')
    var_axs.loglog(variance_gradLoss.cpu().detach().numpy(), color='red')

    print(f"Final values were: {expected_gradLoss[-1].item()} and {variance_gradLoss[-1].item()}")

    avg_fig.savefig(f'img/{FILE}/Expected Variance of Loss.png')
    var_fig.savefig(f'img/{FILE}/Variance of Variance of Loss.png')

