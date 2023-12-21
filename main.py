import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

import sys; sys.path.append('..')
from src.networks.EBM import tiltedpriorEBM
from src.networks.GEN import topdownGenerator
from src.networks.temperedGEN import temperedGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.utils.plot_sample_funcs import generate_sample, save_one_sample, save_grid
from src.utils.diagnostics import plot_hist, plot_pdf, plot_temps

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 20
NUM_BATCHES = 50
NUM_DATA = 5000 # 60000 # Full dataset is 60000 samples
BATCH_SIZE = NUM_DATA // NUM_BATCHES

Z_SAMPLES = 100 # Size of latent Z vector
EMB_OUT_SIZE = 1 # Size of output of EBM
GEN_OUT_CHANNELS = 1 # Size of output of GEN
GEN_FEATURE_DIM = 128 # Feature dimensions of generator
EBM_FEATURE_DIM = 64 # Feature dimensions of EBM

E_LR = 0.0002
G_LR = 0.001

E_STEP = 0.2
G_STEP = 0.1

E_SAMPLE_STEPS = 20
G_SAMPLE_STEPS = 20

p0_SIGMA = 0.15
GENERATOR_SIGMA = 0.1

TEMP_SCHEDULE = 'uniform'
NUM_TEMPS = 10

SAMPLE_BREAK = NUM_EPOCHS // 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for computation.")

# Transforms to apply to dataset. Normalising improves data convergence, numerical stability, and regularisation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset_full = MNIST(root="dataset/", train=True, transform=transform, download=True)
train_dataset = torch.utils.data.Subset(train_dataset_full, range(0, NUM_DATA))  # Use the first NUM_DATA samples
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

Sampler = langevin_sampler(
    p0_sigma=p0_SIGMA, 
    batch_size=BATCH_SIZE, 
    num_latent_samples=Z_SAMPLES, 
    device=device
)

EBMnet = tiltedpriorEBM(
    input_dim=Z_SAMPLES, 
    feature_dim=EBM_FEATURE_DIM,
    output_dim=EMB_OUT_SIZE, 
    p0_sigma=p0_SIGMA, 
    langevin_steps=E_SAMPLE_STEPS, 
    langevin_s=E_STEP
).to(device)

# GENnet = topdownGenerator(
#     input_dim=Z_SAMPLES,
#     feature_dim=GEN_FEATURE_DIM, 
#     output_dim=GEN_OUT_CHANNELS, 
#     sampler=Sampler,
#     lkhood_sigma=GENERATOR_SIGMA, 
#     langevin_steps=G_SAMPLE_STEPS, 
#     langevin_s=G_STEP,
#     device = device
# ).to(device)

GENnet = temperedGenerator(
    input_dim=Z_SAMPLES,
    feature_dim=GEN_FEATURE_DIM, 
    output_dim=GEN_OUT_CHANNELS, 
    sampler=Sampler,
    lkhood_sigma=GENERATOR_SIGMA,
    langevin_steps=G_SAMPLE_STEPS,
    langevin_s=G_STEP,
    num_replicas=NUM_TEMPS,
    temp_schedule_power=1,
    device=device
).to(device)

# File for saving images
print(f"Using {GENnet.__class__.__name__} model.")
FILE = 'Power Posteriors Alt' if GENnet.__class__.__name__ == 'temperedGenerator' else 'Vanilla Pang'

EBMoptimiser = torch.optim.Adam(EBMnet.parameters(), lr=E_LR)
EBMnet.optimiser = EBMoptimiser
GENoptimiser = torch.optim.Adam(GENnet.parameters(), lr=G_LR)
GENnet.optimiser = GENoptimiser

print(f"Training {GENnet.__class__.__name__} model for {NUM_EPOCHS} epochs with {NUM_BATCHES} batches per epoch.")

tqdm_bar = tqdm(range(NUM_EPOCHS))
writer = SummaryWriter(f"runs/{FILE}")

# Get the temperature schedule as a PyTorch tensor
if GENnet.__class__.__name__ == 'temperedGenerator':
    temperatures = GENnet.temp_schedule
else:
    temperatures = ['N/A']

# Initialise plots for expected variance and variance of variances
avg_fig, avg_axs = plt.subplots(1, 1, figsize=(18, 6))
avg_fig.suptitle("Expected Variance of Generated Samples")
avg_axs.set_xlabel("Epoch")
avg_axs.set_ylabel("Expected Variance")

var_fig, var_axs = plt.subplots(1, 1, figsize=(18, 6))
var_fig.suptitle("Variance of Variance of Generated Samples")
var_axs.set_xlabel("Epoch")
var_axs.set_ylabel("Variance of Variance")

# Initialise arrays to store expected variance and variance of variances in zK_GEN for each temperature
avg_var = torch.zeros(NUM_EPOCHS, len(temperatures), device=device)
var_var = torch.zeros(NUM_EPOCHS, len(temperatures), device=device)

stored_samples = torch.zeros(5, 1, 28, 28, device=device)

# Different colour for each temperature
cmap = plt.get_cmap('jet')

for epoch in tqdm_bar:
    EBMtotal_loss = 0
    GENtotal_loss = 0
    sum_batch_avg = torch.zeros(len(temperatures), device=device)
    sum_batch_var = torch.zeros(len(temperatures), device=device)

    for batch_idx, (batch, _) in enumerate(loader): 

        # expected_var, var_var = are the expected variance and variance of variances of zK_GEN samples for each temperature
        lossG, lossE, expected_var_idx, var_var_idx = GENnet.train(batch, EBMnet)
        EBMtotal_loss += lossE
        GENtotal_loss += lossG
        sum_batch_avg += torch.tensor(expected_var_idx, device=device).detach()
        sum_batch_var += torch.tensor(var_var_idx, device=device).detach()
        tqdm_bar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}; Batch {batch_idx}/{NUM_BATCHES}; EBM-Loss: {EBMtotal_loss / (batch_idx + 1):.4f} GEN-Loss: {GENtotal_loss / (batch_idx + 1):.4f}")

    avg_var[epoch] = sum_batch_avg / NUM_BATCHES # Average expected variance of zK_GEN for each temperature
    var_var[epoch] = sum_batch_var / NUM_BATCHES # Average variance of variances of zK_GEN for each temperature

    if (epoch % SAMPLE_BREAK == 0 or epoch == NUM_EPOCHS):
        generated_data = generate_sample(GENnet, EBMnet).reshape(-1, 1, 28, 28)
        img_grid = torchvision.utils.make_grid(generated_data, normalize=True)

        writer.add_image(f"Generated Samples -- {FILE} Model", img_grid, global_step=epoch)

        # Stores 5 generated samples
        generated_data = generate_sample(GENnet, EBMnet).reshape(-1, 1, 28, 28)[0:5]
        stored_samples = torch.cat((stored_samples, generated_data), dim=0)

        # # Plot the expected variance and variance of variances
        # if GENnet.__class__.__name__ == 'temperedGenerator':
        #     plot_temps(Sampler, EBMnet, GENnet, batch.to(device), epoch=epoch, num_plots=5)

        # Save the gride of generated samples
        save_grid(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], epoch=epoch, file=FILE, num_images=32)

writer.close()

# Plot the expected variance and variance of variances
for idx, temp in enumerate(temperatures):
    avg_axs.plot(avg_var[:, idx].cpu().detach().numpy(), label=f"Temperature = {temp}", color=cmap(idx / len(temperatures)))
    var_axs.plot(var_var[:, idx].cpu().detach().numpy(), label=f"Temperature = {temp}", color=cmap(idx / len(temperatures)))

avg_axs.legend()
avg_fig.savefig(f'img/{FILE}/Expected Variance.png')
var_axs.legend()
var_fig.savefig(f'img/{FILE}/Variance of Variance.png')

# Plot the final generated image/grid
save_one_sample(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE)
save_grid(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE, num_images=32)

# Plot the stored grid
save_grid(stored_samples, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE, num_images=stored_samples.size(0), name='Evolution of Samples')

# Diagnostics
X = batch.to(device)
plot_hist(Sampler, EBMnet, GENnet, X, file=FILE)
plot_pdf(Sampler, EBMnet, GENnet, batch.to(device), file=FILE)

# Sampler.batch_size = 100
# X = train_dataset_full.data[:100].to(device).unsqueeze(1).float()
# plot_pdf(Sampler, EBMnet, GENnet, X.float(), file=FILE)

