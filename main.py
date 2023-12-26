import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

import sys; sys.path.append('..')
from src.networks.EBM import tiltedpriorEBM
from src.networks.GEN import topdownGenerator
from src.networks.temperedGEN import temperedGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.utils.plotting_functions import generate_sample, save_one_sample, save_grid, plot_posterior_metrics, plot_gradLoss_metrics
from src.utils.diagnostics import plot_hist, plot_pdf, plot_temps

# Set plot styling
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 200
NUM_BATCHES = 100
NUM_DATA = 2500 # 60000 # Full MNIST dataset is 60000 samples
BATCH_SIZE = NUM_DATA // NUM_BATCHES

Z_SAMPLES = 100 # Size of latent Z vector
EMB_OUT_SIZE = 1 # Size of output of EBM
GEN_FEATURE_DIM = 128 # Feature dimensions of generator
EBM_FEATURE_DIM = 64 # Feature dimensions of EBM

CHANNELS = 3 # Size of output of GEN
IMAGE_DIM = 64

E_LR = 0.0002
G_LR = 0.001

E_STEP = 0.2
G_STEP = 0.1

E_SAMPLE_STEPS = 40
G_SAMPLE_STEPS = 40

p0_SIGMA = 0.15
GENERATOR_SIGMA = 0.1

TEMP_SCHEDULE = 'uniform'
NUM_TEMPS = 10

SAMPLE_BREAK = NUM_EPOCHS // 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for computation.")

# Transforms to apply to dataset. Normalising improves data convergence, numerical stability, and regularisation.
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the images to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
])

# Load the CelebA dataset
train_dataset_full = torchvision.datasets.CelebA(
    root="dataset/",
    split="train",
    transform=transform,
    download=True
)
train_dataset = torch.utils.data.Subset(train_dataset_full, range(0, NUM_DATA))  # Use the first NUM_DATA samples
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate models and MCMC sampler
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

GENnet = topdownGenerator(
    input_dim=Z_SAMPLES,
    feature_dim=GEN_FEATURE_DIM, 
    output_dim=CHANNELS, 
    sampler=Sampler,
    lkhood_sigma=GENERATOR_SIGMA, 
    langevin_steps=G_SAMPLE_STEPS, 
    langevin_s=G_STEP,
    device = device
).to(device)

# GENnet = temperedGenerator(
#     input_dim=Z_SAMPLES,
#     feature_dim=GEN_FEATURE_DIM, 
#     output_dim=GEN_OUT_CHANNELS, 
#     sampler=Sampler,
#     lkhood_sigma=GENERATOR_SIGMA,
#     langevin_steps=G_SAMPLE_STEPS,
#     langevin_s=G_STEP,
#     num_replicas=NUM_TEMPS,
#     temp_schedule_power=1,
#     device=device
# ).to(device)

# File for saving images
print(f"Using {GENnet.__class__.__name__} model.")
FILE = 'Power Posteriors Alt' if GENnet.__class__.__name__ == 'temperedGenerator' else 'Vanilla Pang'

# Initialise optimisers
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

# Initialise arrays to store expected variance and variance of variances in zK_GEN for each temperature
avg_var_posterior = torch.zeros(NUM_EPOCHS, len(temperatures), device=device)
var_var_posterior = torch.zeros(NUM_EPOCHS, len(temperatures), device=device)

# Initialise array to store generated samples to plot a nice sample evolution figure
stored_samples = torch.zeros(5, CHANNELS, IMAGE_DIM, IMAGE_DIM, device=device)

# Different colour for each temperature in the temp variance plot
cmap = plt.get_cmap('jet')

# Initialise arrays to store expected/variance in variances of the loss function evaluations
expected_gradloss = torch.zeros(NUM_EPOCHS, device=device)
variance_gradloss = torch.zeros(NUM_EPOCHS, device=device)

# Initialise array to store FID scores
FID_scores = []
fid = FrechetInceptionDistance(feature=64).to(device)

for epoch in tqdm_bar:
    # Running loss
    EBMtotal_loss = 0
    GENtotal_loss = 0

    # Initialise arrays to store expected/variance in variances of zK_GEN samples for each temperature in this epoch
    sum_batch_avg = torch.zeros(len(temperatures), device=device)
    sum_batch_var = torch.zeros(len(temperatures), device=device)

    lossLoss_avg = torch.zeros(NUM_BATCHES, device=device)
    lossLoss_var = torch.zeros(NUM_BATCHES, device=device)

    for batch_idx, (batch, _) in enumerate(loader): 

        # Train the models. Get the losses, posterior sample variances, and gradients
        losses, posterior_metrics, loss_gradient_metrics = GENnet.train(batch, EBMnet)

        # Update running losses
        GENtotal_loss += losses[0]
        EBMtotal_loss += losses[1]

        # Update running averages/variances of posterior sample variances
        sum_batch_avg += torch.tensor(posterior_metrics[0], device=device).clone().detach()
        sum_batch_var += torch.tensor(posterior_metrics[1], device=device).clone().detach()

        # Update running averages/variances of loss function evaluations
        lossLoss_avg[batch_idx] = loss_gradient_metrics[0]
        lossLoss_var[batch_idx] = loss_gradient_metrics[1]
        
        tqdm_bar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}; Batch {batch_idx}/{NUM_BATCHES}; EBM-Loss: {EBMtotal_loss / (batch_idx + 1):.4f} GEN-Loss: {GENtotal_loss / (batch_idx + 1):.4f}")

        # For speedy testing - comment out when serious
        # if batch_idx > 5:
        #     break

    avg_var_posterior[epoch] = sum_batch_avg / NUM_BATCHES # Batch average expected variance of zK_GEN for each temperature
    var_var_posterior[epoch] = sum_batch_var / NUM_BATCHES # Batch average variance of variances of zK_GEN for each temperature

    expected_gradloss[epoch] = torch.mean(lossLoss_var) # Batch expected variance of zK_GEN for each temperature
    variance_gradloss[epoch] = torch.var(lossLoss_var) # Batch variance of variances of zK_GEN for each temperature

    if (epoch % SAMPLE_BREAK == 0 or epoch == NUM_EPOCHS):
        generated_data = generate_sample(GENnet, EBMnet).reshape(-1, CHANNELS, IMAGE_DIM, IMAGE_DIM)
        img_grid = torchvision.utils.make_grid(generated_data, normalize=True)

        writer.add_image(f"Generated Samples -- {FILE} Model", img_grid, global_step=epoch)

        # Stores 5 generated samples
        stored_samples = torch.cat((stored_samples, generated_data[0:9]), dim=0)

        # Calculate FID score
        fid.update(batch.to(device).to(torch.uint8).reshape(-1, CHANNELS, IMAGE_DIM, IMAGE_DIM), real=True)
        fid.update(generated_data.to(torch.uint8), real=False)
        FID_scores.append((epoch,fid.compute().cpu().detach().numpy()))

        # # Plot the expected variance and variance of variances
        # if GENnet.__class__.__name__ == 'temperedGenerator':
        #     plot_temps(Sampler, EBMnet, GENnet, batch.to(device), epoch=epoch, num_plots=5)

writer.close()

# Plot the expected/variance in variance of zK_GEN
plot_posterior_metrics(avg_var_posterior, var_var_posterior, temperatures, cmap, FILE)

# Plot the expected/variance in variance of loss
plot_gradLoss_metrics(expected_gradloss, variance_gradloss, FILE)

# Plot all the final generated images in a grid
#save_one_sample(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE)
save_grid(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], epoch=NUM_EPOCHS, file=FILE, num_images=32)

# Plot the stored grid to showcase the evolution
save_grid(stored_samples, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], epoch=NUM_EPOCHS, file=FILE, num_images=stored_samples.size(0), name='Evolution of Samples')

# Diagnostics
X = batch.to(device)
plot_hist(Sampler, EBMnet, GENnet, X, file=FILE)
plot_pdf(Sampler, EBMnet, GENnet, batch.to(device), file=FILE)

plt.figure()
plt.plot(FID_scores)
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.title('FID Score Evolution')
plt.savefig(f'../figures/{FILE}/FID Score Evolution.png', dpi=300)


