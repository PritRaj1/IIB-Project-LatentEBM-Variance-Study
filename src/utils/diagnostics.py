import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

def plot_hist(Sampler, EBMnet, GENnet, x, file='Vanilla Pang'):
    """
    Function to plot the histograms of the prior and posterior distributions

    Parameters:
    - Sampler: object used for MCMC sampling
    - EBMnet: neural network used for energy-based modeling
    - GENnet: neural network used for generative modeling
    - x: torch.Tensor of shape (batch_size, 1, 28, 28) containing the images
    """
    print("Plotting histograms...")

    # Sample from the noise prior distribution
    z0 = Sampler.sample_p0() 

    # Sample from the posterior distribution and plot the results
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_GEN = Sampler.get_sample(zK_EBM, x, GENnet, EBMnet)

    # Mean along the dimension NUM_Z
    zK_EBM_mean = torch.mean(zK_EBM, dim=1).squeeze()
    zK_GEN_mean = torch.mean(zK_GEN, dim=1).squeeze()

    # Normalise
    zK_EBM_mean = (zK_EBM_mean - zK_EBM_mean.mean()) / zK_EBM_mean.std()
    zK_GEN_mean = (zK_GEN_mean - zK_GEN_mean.mean()) / zK_GEN_mean.std()

    # Create a new figure with subplots
    plt.figure(figsize=(12, 6))

    # Plot the histograms on the first subplot
    sns.histplot(zK_EBM_mean.detach().cpu().numpy(), bins=20, color='red', kde=False, label=r'$p_\alpha(z)$ from EBM')
    sns.histplot(zK_GEN_mean.detach().cpu().numpy(), bins=20, color='blue', kde=False, label=r'$p_\theta(z|x)$ from GEN')

    plt.title(r'Histogram of Prior-posterior Matching')

    # Add legends and save the figure
    plt.legend()
    plt.legend()
    plt.savefig(f'img/{file}/Histplot.png')

def plot_pdf(Sampler, EBMnet, GENnet, X, file='Vanilla Pang'):
    """
    Function to plot the PDFs of the pixel instensities of the real and generated images

    Parameters:
    - Sampler: object used for MCMC sampling
    - EBMnet: neural network used for energy-based modeling
    - GENnet: neural network used for generative modeling
    - X: torch.Tensor of shape (batch_size, 1, 28, 28) containing the images
    """

    print("Plotting PDFs...")
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Kernel Density Estimation of Data and Generated Distributions")

    # Sample from the prior distribution and plot the results
    z0 = Sampler.sample_p0() 

    # Sample from the posterior distribution and plot the results
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_GEN = Sampler.get_sample(zK_EBM, X, GENnet, EBMnet)

    with torch.no_grad():
        X_pred1 = torch.mean(GENnet(zK_EBM.detach()), dim=1).squeeze()
        X_pred2 = torch.mean(GENnet(zK_GEN.detach()), dim=1).squeeze()

    sns.kdeplot(X.view(-1).cpu().numpy(), levels=50, ax=axs[0])
    axs[0].set_title(r"$p(x)$ -- Real Distribution")

    # Generate the heatmap for the predicted distributions on the second subplot
    sns.kdeplot(X_pred1.view(-1).cpu().numpy(), fill=True, cmap='Reds', levels=50, ax=axs[1])
    axs[1].set_title(r"$p_\alpha(x|z)$ -- z sampled from $p_\alpha(z)$")

    # Generate the heatmap for the predicted distributions on the third subplot
    sns.kdeplot(X_pred2.view(-1).cpu().numpy(), fill=True, cmap='Reds', levels=50, ax=axs[2])
    axs[2].set_title(r"$p_\theta(x|z)$ -- z sampled from $p_\theta(z|x)$")

    plt.title(r'Kernel Density Estimation of Data and Generated Pixel Intensities')

    # Save the figure as a PNG file
    fig.savefig(f'img/{file}/PDFplot.png')

def plot_temps(Sampler, EBMnet, GENnet, x, epoch, num_plots=5):
    FILE = 'Power Posteriors Alt'

    # Create subplots
    fig, axs = plt.subplots(1, num_plots, figsize=(18, 6))
    fig.suptitle(f"Posterior Density Plots for Different Temperatures -- Epoch {epoch}")

    # Every t_index-th temperature is plotted
    t_index = len(GENnet.temp_schedule) // num_plots
    temps = GENnet.temp_schedule[1::t_index]

    # Sample from the noise prior distribution
    z0 = Sampler.sample_p0() 

    # Sample from the posterior distribution and plot the results
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_EBM_mean = torch.mean(zK_EBM, dim=1).squeeze()
    zK_EBM_mean = (zK_EBM_mean - zK_EBM_mean.mean()) / zK_EBM_mean.std()

    for i, temp in enumerate(temps):
        # Set replica temperature
        GENnet.current_temp = temp

        # Sample from the posterior distribution and plot the results
        zK_GEN = Sampler.get_sample(zK_EBM, x, GENnet, EBMnet)

        # Mean along the dimension 1
        zK_GEN_mean = torch.mean(zK_GEN, dim=1).squeeze()
        zK_GEN_mean = (zK_GEN_mean - zK_GEN_mean.mean()) / zK_GEN_mean.std()

        # Plot the histograms on the first subplot
        sns.axes_style("darkgrid")
        sns.histplot(zK_EBM_mean.detach().cpu().numpy(), bins=20, color='red', kde=False, label=r'$p_\alpha(z)$ from EBM', ax=axs[i])
        sns.histplot(zK_GEN_mean.detach().cpu().numpy(), bins=20, color='blue', kde=False, label=r'$p_\theta(z|x)$ from GEN', ax=axs[i])

        axs[i].set_title(f"Temperature: {temp:.3f}")

    # Add legends and save the figure
    plt.legend()
    plt.savefig(f'img/{FILE}/Temps Epoch {epoch}.png')

