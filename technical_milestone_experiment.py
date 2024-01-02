"""
File to collect variances in grad loss for different setups, as part 
of the technical milestone report. The variances are stored as .pth objects
ready for plotting in TMR_plots.py
"""


import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from matplotlib import rc
import seaborn as sns

import sys; sys.path.append('..')
from src.networks.EBM import tiltedpriorEBM
from src.networks.GEN import topdownGenerator
from src.networks.temperedGEN import temperedGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.utils.plotting_functions import generate_sample, save_grid
from src.utils.helper_functions import create_results_directory

# Set plot styling
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=14)
rc('text', usetex=True)
sns.set_style("darkgrid")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 100
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

# List of temperature powers, 0 = vaniall generator model
temp_powers = [3,4,5]#[0, 1, 2, 3, 4, 5]

for p in temp_powers:
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
    
    if p == 0:
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

    else:
        GENnet = temperedGenerator(
            input_dim=Z_SAMPLES,
            feature_dim=GEN_FEATURE_DIM, 
            output_dim=CHANNELS, 
            sampler=Sampler,
            lkhood_sigma=GENERATOR_SIGMA, 
            langevin_steps=G_SAMPLE_STEPS, 
            langevin_s=G_STEP,
            num_replicas=NUM_TEMPS,
            temp_schedule_power=p, # Varying this
            device = device
        ).to(device)

    # File for saving images
    print(f"Using {GENnet.__class__.__name__} model.")
    FILE = 'Power Posteriors Alt' if GENnet.__class__.__name__ == 'temperedGenerator' else 'Vanilla Pang'
    SUBFILE = f'p={p}' if GENnet.__class__.__name__ == 'temperedGenerator' else 'Vanilla'

    # Make sure files exist, if not create them
    create_results_directory(FILE, SUBFILE)

    # Initialise optimisers
    optimiser_E = torch.optim.Adam(EBMnet.parameters(), lr=E_LR)
    EBMnet.optimiser = optimiser_E
    optimiser_G = torch.optim.Adam(GENnet.parameters(), lr=G_LR)
    GENnet.optimiser = optimiser_G

    print(f"Training {GENnet.__class__.__name__}, p={p}, model for {NUM_EPOCHS} epochs with {NUM_BATCHES} batches per epoch.")

    # Initialise progress bar 
    tqdm_bar = tqdm(range(NUM_EPOCHS))

    # Initialise array to store generated samples to plot a nice sample evolution figure
    stored_samples = torch.zeros(0, CHANNELS, IMAGE_DIM, IMAGE_DIM, device=device)

    # Initialise arrays to store avg loss
    total_avg_loss = torch.zeros(NUM_EPOCHS, device=device)

    # Initialise arrays to store expected/variance in variances of the loss function evaluations
    expected_gradloss = torch.zeros(NUM_EPOCHS, device=device)
    variance_gradloss = torch.zeros(NUM_EPOCHS, device=device)

    # Initialise array to store FID scores
    FID_scores = torch.zeros(NUM_EPOCHS, device=device)
    fid = FrechetInceptionDistance(feature=64).to(device) # FID metric

    for epoch in tqdm_bar:
        # Running loss
        EBMtotal_loss = 0
        GENtotal_loss = 0

        gradLoss_avg = torch.zeros(NUM_BATCHES, device=device)
        gradLoss_var = torch.zeros(NUM_BATCHES, device=device)

        for batch_idx, (batch, _) in enumerate(loader): 
            # Train the models. Get the losses, posterior sample variances, and gradients
            losses, posterior_metrics, loss_gradient_metrics = GENnet.train(batch, EBMnet)

            # Update running losses
            GENtotal_loss += losses[0]
            EBMtotal_loss += losses[1]

            # Update running averages/variances of loss function evaluations
            gradLoss_avg[batch_idx] = loss_gradient_metrics[0]
            gradLoss_var[batch_idx] = loss_gradient_metrics[1]
            
            tqdm_bar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}; Batch {batch_idx}/{NUM_BATCHES}; EBM-Loss: {EBMtotal_loss / (batch_idx + 1):.4f} GEN-Loss: {GENtotal_loss / (batch_idx + 1):.4f}")

        # Track the loss
        total_avg_loss[epoch] = (GENtotal_loss + EBMtotal_loss) / NUM_BATCHES

        # Loss function evaluations
        expected_gradloss[epoch] = torch.mean(gradLoss_var) # Batch expected variance of zK_GEN for each temperature
        variance_gradloss[epoch] = torch.var(gradLoss_var) # Batch variance of variances of zK_GEN for each temperature

        generated_data = generate_sample(GENnet, EBMnet).reshape(-1, CHANNELS, IMAGE_DIM, IMAGE_DIM)

        # Stores 10 generated images
        stored_samples = torch.cat((stored_samples, generated_data[0:2]), dim=0)

        # Calculate FID score
        fid.update(batch.to(device).to(torch.uint8).reshape(-1, CHANNELS, IMAGE_DIM, IMAGE_DIM), real=True)
        fid.update(generated_data.to(torch.uint8), real=False)
        FID_scores[epoch] = fid.compute()
    
    # Save the model
    torch.save(GENnet.state_dict(), f"results/{FILE}/{SUBFILE}/GENnet.pth")
    torch.save(EBMnet.state_dict(), f"results/{FILE}/{SUBFILE}/EBMnet.pth")

    # Save the evolving samples
    save_grid(stored_samples, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA], epoch=NUM_EPOCHS, file=FILE, subfile=SUBFILE, num_images=stored_samples.size(0), name='Evolution of Samples')

    # Save the results with torch.save
    torch.save(total_avg_loss, f"results/{FILE}/{SUBFILE}/total_avg_loss.pth")
    torch.save(expected_gradloss, f"results/{FILE}/{SUBFILE}/expected_gradloss.pth")
    torch.save(variance_gradloss, f"results/{FILE}/{SUBFILE}/variance_gradloss.pth")
    torch.save(FID_scores, f"results/{FILE}/{SUBFILE}/FID_scores.pth")

    # Free up memory
    del GENnet
    del EBMnet
    del optimiser_E
    del optimiser_G
    del stored_samples
    del total_avg_loss
    del expected_gradloss
    del variance_gradloss
    del FID_scores
    del fid
    torch.cuda.empty_cache()




    

