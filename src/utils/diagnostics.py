import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_hist(Sampler, EBMnet, GENnet, x):
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

    # Mean along the dimension 1
    zK_EBM_mean = torch.mean(zK_EBM, dim=1).squeeze()
    zK_GEN_mean = torch.mean(zK_GEN, dim=1).squeeze()

    # Create a new figure with subplots
    plt.figure(figsize=(12, 6))

    # Plot the histograms on the first subplot
    sns.axes_style("darkgrid")
    sns.histplot(zK_EBM_mean.detach().cpu().numpy(), color='red', kde=True, label=r'$p_\alpha(z)$ from EBM')
    sns.histplot(zK_GEN_mean.detach().cpu().numpy(), color='blue', kde=True, label=r'$p_\theta(z|x)$ from GEN')

    plt.title(r'Histogram of prior-posterior matching')

    # Add legends and save the figure
    plt.legend()
    plt.legend()
    plt.savefig('img/Histplot.png')

def plot_pdf(Sampler, EBMnet, GENnet, X):
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
    fig.savefig('img/PDFplot.png')
