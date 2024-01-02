"""
This file is used for profiling. Given the computational requirements of the model, 
(especially with thermodynamic integration involved), it is important to monitor
the memory usage and compute time of the model. This file is used to profile the
model and generate a tensorboard log of the results.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import sys; sys.path.append('..')
from src.networks.EBM import tiltedpriorEBM
from src.networks.GEN import topdownGenerator
from src.networks.temperedGEN import temperedGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.utils.plotting_functions import generate_sample, save_one_sample, save_final_grid
from src.utils.diagnostics import plot_hist, plot_pdf


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32

Z_SAMPLES = 100 # Size of latent Z vector
EMB_OUT_SIZE = 3 # Size of output of EBM
GEN_OUT_CHANNELS = 3 # Size of output of GEN
GEN_FEATURE_DIM = 64 # Feature dimensions of generator
EBM_FEATURE_DIM = 250 # Feature dimensions of EBM

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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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

# Initialise the model
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
    output_dim=GEN_OUT_CHANNELS, 
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
#     temp_schedule_power=1
# ).to(device)

# File for saving images
print(f"Using {GENnet.__class__.__name__} model.")
FILE = 'Power Posteriors Alt' if GENnet.__class__.__name__ == 'temperedGenerator' else 'Vanilla Pang'

# Initialise optimisers
EBMoptimiser = torch.optim.Adam(EBMnet.parameters(), lr=E_LR)
EBMnet.optimiser = EBMoptimiser
GENoptimiser = torch.optim.Adam(GENnet.parameters(), lr=G_LR)
GENnet.optimiser = GENoptimiser

tqdm_bar = tqdm(range(NUM_EPOCHS))
writer = SummaryWriter(f"runs/{FILE}")

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./runs/{FILE}/profilerlogs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for epoch in tqdm_bar:
        EBMtotal_loss = 0
        GENtotal_loss = 0

        for batch_idx, (batch, _) in enumerate(loader): 

            losses, _, _ = GENnet.train(batch, EBMnet)
            EBMtotal_loss += losses[0]
            GENtotal_loss += losses[1]

            prof.step()
            if batch_idx >= 5:
                break
        
        tqdm_bar.set_description(f"Epoch {epoch}: EBM-Loss: {EBMtotal_loss / (BATCH_SIZE):.4f} GEN-Loss: {GENtotal_loss / (BATCH_SIZE):.4f}")

        if (epoch % SAMPLE_BREAK == 0 or epoch == NUM_EPOCHS):
            generated_data = generate_sample(GENnet, EBMnet).reshape(-1, 1, 28, 28)
            img_grid = torchvision.utils.make_grid(generated_data, normalize=True)

            writer.add_image(f"Generated Samples -- {FILE} Model", img_grid, global_step=epoch)

writer.close()

# Plot the final generated image/grid
save_one_sample(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE)
save_final_grid(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], file=FILE)

# Diagnostics
plot_hist(Sampler, EBMnet, GENnet, x, file=FILE)
Sampler.batch_size = 100
X = train_dataset.data[:100].to(device).unsqueeze(1).float()
plot_pdf(Sampler, EBMnet, GENnet, X.float(), file=FILE)


