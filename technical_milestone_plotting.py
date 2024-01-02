import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styling
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

# Get results
expected_loss = [
    torch.load('results/Vanilla Pang/Vanilla/total_avg_loss.pth'),
    torch.load('results/Power Posteriors Alt/p=1/total_avg_loss.pth'),
    torch.load('results/Power Posteriors Alt/p=2/total_avg_loss.pth'),
    torch.load('results/Power Posteriors Alt/p=3/total_avg_loss.pth'),
    torch.load('results/Power Posteriors Alt/p=4/total_avg_loss.pth'),
    torch.load('results/Power Posteriors Alt/p=5/total_avg_loss.pth')
]

variances =[
    torch.load('results/Vanilla Pang/Vanilla/variance_gradloss.pth'),
    torch.load('results/Power Posteriors Alt/p=1/variance_gradloss.pth'),
    torch.load('results/Power Posteriors Alt/p=2/variance_gradloss.pth'),
    torch.load('results/Power Posteriors Alt/p=3/variance_gradloss.pth'),
    torch.load('results/Power Posteriors Alt/p=4/variance_gradloss.pth'),
    torch.load('results/Power Posteriors Alt/p=5/variance_gradloss.pth')
]

FIDs = [
    torch.load('results/Vanilla Pang/Vanilla/FID_scores.pth'),
    torch.load('results/Power Posteriors Alt/p=1/FID_scores.pth'),
    torch.load('results/Power Posteriors Alt/p=2/FID_scores.pth'),
    torch.load('results/Power Posteriors Alt/p=3/FID_scores.pth'),
    torch.load('results/Power Posteriors Alt/p=4/FID_scores.pth'),
    torch.load('results/Power Posteriors Alt/p=5/FID_scores.pth')
]

plot_titles=[
    r"\mathrm{E}_{p(\mathbf{x})}\left[-\log\left(p(\mathbf{\tilde{x}}|\mathbf{\theta})\right)\right]", # loss
    r"\mathrm{Var}_{p(\mathbf{x})}\left[mathrm{Var}_{p(\mathbf{x})}\left[\nabla_{\mathbf{\theta}}-\log\left(p(\mathbf{\tilde{x}}|\mathbf{\theta})\right)\right]\right]", # variance
    r"\mathrm{FID}_{p(\mathbf{x})}\left[p(\mathbf{\tilde{x}}|\mathbf{\theta})\right]" # FID
]

plot_labels=[
    "Vanilla",
    r"$p=1$",
    r"$p=2$",
    r"$p=3$",
    r"$p=4$",
    r"$p=5$"
]

file_names=[
    "loss",
    "variance",
    "FID"
]

def plot_results(results, plot_title, plot_labels, save_name):
    """
    Plot results from technical milestone experiments.
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(results)):
        plt.loglog(results[i], label=plot_labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title(plot_title)
    plt.savefig(save_name)
    plt.close()

for result_idx, results in enumerate([expected_loss, variances, FIDs]):
    plot_results(results, plot_titles[i], plot_labels, f"results/{file_names[i]}.png")


        
