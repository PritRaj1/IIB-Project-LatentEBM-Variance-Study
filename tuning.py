import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from torchvision import transforms
import optuna
from matplotlib import rc
import seaborn as sns
from torchmetrics.image.fid import FrechetInceptionDistance

import sys; sys.path.append('..')
from src.networks.EBM import tiltedpriorEBM
from src.networks.GEN import topdownGenerator
from src.networks.temperedGEN import temperedGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.utils.plotting_functions import generate_sample

# Constant hyperparameters
NUM_EPOCHS = 200
NUM_BATCHES = 100
NUM_DATA = 2500 # 60000 # Full MNIST dataset is 60000 samples
BATCH_SIZE = NUM_DATA // NUM_BATCHES

Z_SAMPLES = 100 # Size of latent Z vector
EMB_OUT_SIZE = 1 # Size of output of EBM
GEN_OUT_CHANNELS = 1 # Size of output of GEN
GEN_FEATURE_DIM = 128 # Feature dimensions of generator
EBM_FEATURE_DIM = 64 # Feature dimensions of EBM

E_SAMPLE_STEPS = 40
G_SAMPLE_STEPS = 40

CHANNELS = 3 # Size of output of GEN
IMAGE_DIM = 64

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

fid = FrechetInceptionDistance(feature=64).to(device) # FID metric

def objective(trial):
    EBM_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    GEN_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    EBM_langevin_step = trial.suggest_float('langevin_step', 0.01, 0.5)
    GEN_langevin_step = trial.suggest_float('langevin_step', 0.01, 0.5)
    prior_sigma = trial.suggest_float('prior_sigma', 0.01, 0.5)
    generator_sigma = trial.suggest_float('generator_sigma', 0.01, 0.5)

    # Instantiate models and MCMC sampler
    Sampler = langevin_sampler(
        p0_sigma=prior_sigma, 
        batch_size=BATCH_SIZE, 
        num_latent_samples=Z_SAMPLES, 
        device=device
    )

    EBMnet = tiltedpriorEBM(
        input_dim=Z_SAMPLES, 
        feature_dim=EBM_FEATURE_DIM,
        output_dim=EMB_OUT_SIZE, 
        p0_sigma=prior_sigma, 
        langevin_steps=E_SAMPLE_STEPS, 
        langevin_s=EBM_langevin_step
    ).to(device)

    GENnet = topdownGenerator(
        input_dim=Z_SAMPLES,
        feature_dim=GEN_FEATURE_DIM, 
        output_dim=GEN_OUT_CHANNELS, 
        sampler=Sampler,
        lkhood_sigma=generator_sigma, 
        langevin_steps=G_SAMPLE_STEPS, 
        langevin_s=GEN_langevin_step,
        device = device
    ).to(device)

    # Initialise optimisers
    EBMoptimiser = torch.optim.Adam(EBMnet.parameters(), lr=EBM_learning_rate)
    EBMnet.optimiser = EBMoptimiser
    GENoptimiser = torch.optim.Adam(GENnet.parameters(), lr=GEN_learning_rate)
    GENnet.optimiser = GENoptimiser

    for epoch in NUM_EPOCHS:

        for batch_idx, (batch, _) in enumerate(loader): 

            # Train the models. Get the losses, posterior sample variances, and gradients
            losses, posterior_metrics, loss_gradient_metrics = GENnet.train(batch, EBMnet)
    
    generated_data = generate_sample(GENnet, EBMnet).reshape(-1, 1, 28, 28)

    # Calculate FID score
    fid.update(batch.to(device).to(torch.uint8).reshape(-1, CHANNELS, IMAGE_DIM, IMAGE_DIM), real=True)
    fid.update(generated_data.to(torch.uint8), real=False)
    FID = fid.compute().cpu().detach().numpy()

    return FID


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000) 
print(study.best_trial)
print(study.best_params)
print(study.best_value)

optuna.visualization.plot_optimization_history(study).show()