import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_hist(Sampler, EBMnet, GENnet, x):
    # Sample from the noise prior distribution
    z0 = Sampler.sample_p0() 

    # Sample from the posterior distribution and plot the results
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_GEN = Sampler.get_sample(zK_EBM, x, GENnet, EBMnet)

    # Mean along the final dimension
    zK_EBM_mean = torch.mean(zK_EBM, dim=-1)
    zK_GEN_mean = torch.mean(zK_GEN, dim=-1)

    # Create a new figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the histograms on the first subplot
    sns.axes_style("darkgrid")
    sns.histplot(zK_EBM_mean.detach().cpu().numpy(), color='red', kde=True, label=r'$p_\alpha(z)$ from EBM', ax=axs[0])
    axs[0].set_title('Histogram of $p_\alpha(z)$ from EBM')

    sns.histplot(zK_GEN_mean.detach().cpu().numpy(), color='blue', kde=True, label=r'$p_\theta(z|x)$ from GEN', ax=axs[1])
    axs[1].set_title('Histogram of $p_\theta(z|x)$ from GEN')

    # Add legends and save the figure
    axs[0].legend()
    axs[1].legend()
    plt.savefig('img/Histplot.png')

def plot_pdf(Sampler, EBMnet, GENnet, X, data_dim):
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Kernel Density Estimation of Generated Distribution")

    Sampler.batch_size = X.shape[0]

    # Sample from the prior distribution and plot the results
    z0 = Sampler.sample_p0() 

    # Sample from the posterior distribution and plot the results
    zK_EBM = Sampler.get_sample(z0, None, EBMnet, None)
    zK_GEN = Sampler.get_sample(zK_EBM, X, GENnet, EBMnet)

    X_pred1 = GENnet(zK_EBM.detach())
    X_pred2 = GENnet(zK_GEN.detach())

    # Reshape X_pred1 and X_pred2 to match the original data dimensions
    X_pred1 = X_pred1.reshape(X.shape[0], -1)
    X_pred2 = X_pred2.reshape(X.shape[0], -1)

    # Generate the heatmap for the predicted distributions on the first subplot
    df1 = pd.DataFrame({'x1': X_pred1[:, 0].detach().cpu().flatten(), 'x2': X_pred1[:, 1].detach().cpu().flatten()})
    sns.kdeplot(data=df1, x='x1', y='x2', fill=True, cmap='Reds', levels=50, ax=axs[0])
    axs[0].set_title(r'Generated Dist. -- z sampled from $p_\alpha(z)$')

    # Generate the heatmap for the predicted distributions on the second subplot
    df2 = pd.DataFrame({'x1': X_pred2[:, 0].detach().cpu().flatten(), 'x2': X_pred2[:, 1].detach().cpu().flatten()})
    sns.kdeplot(data=df2, x='x1', y='x2', fill=True, cmap='Reds', levels=50, ax=axs[1])
    axs[1].set_title(r'Generated Dist. -- z sampled from $p_\theta(z|x)$')

    # Save the figure as a PNG file
    fig.savefig('img/PDFplot.png')
