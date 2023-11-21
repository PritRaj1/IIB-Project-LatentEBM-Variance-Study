import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons, make_blobs
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import sys; sys.path.append('..')
from src.models.EBM import tiltedpriorEBM
from src.models.GEN import topdownGenerator
from src.MCMC_Samplers.langevin import langevin_sampler
from src.loss_functions.EBM_loss_fn import EBM_loss
from src.loss_functions.GEN_loss_fn import generator_loss
from src.pipeline import train_step, generate_sample, save_final_sample
from src.utils.diagnostics import plot_hist, plot_pdf


device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 300
BATCH_SIZE = 32


Z_SAMPLES = 100 # Size of latent Z vector
EMB_OUT_SIZE = 1 # Size of output of EBM
GEN_OUT_CHANNELS = 3 # Size of output of GEN
GEN_FEATURE_DIM = 64 # Feature dimensions of generator
EBM_FEATURE_DIM = 200 # Feature dimensions of EBM


E_LR = 0.00002
G_LR = 0.0001

E_STEP = 0.2
G_STEP = 0.1

E_SAMPLE_STEPS = 80
G_SAMPLE_STEPS = 80

p0_SIGMA = 1
GENERATOR_SIGMA = 0.3

SAMPLE_BREAK = NUM_EPOCHS // 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for computation.")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32

# Transforms to apply to dataset. Normalising improves data convergence, numerical stability, and regularisation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
data_dim = train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]

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
    lkhood_sigma=GENERATOR_SIGMA, 
    langevin_steps=G_SAMPLE_STEPS, 
    langevin_s=G_STEP
).to(device)

lossE_fn = EBM_loss
lossG_fn = generator_loss

EBMoptimiser = torch.optim.Adam(EBMnet.parameters(), lr=E_LR)
GENoptimiser = torch.optim.Adam(GENnet.parameters(), lr=G_LR)

tqdm_bar = tqdm(range(NUM_EPOCHS))
writer = SummaryWriter(f"runs/Vanilla_Pang_Model")

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/VanillaEBM/profilerlogs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for epoch in tqdm_bar:
        EBMtotal_loss = 0
        GENtotal_loss = 0

        for batch_idx, (batch, _) in enumerate(loader): 
            x = batch.to(device)

            lossE, lossG = train_step(
                x, 
                GENnet, 
                EBMnet, 
                GENoptimiser, 
                EBMoptimiser, 
                Sampler, 
                lossG_fn, 
                lossE_fn)
            
            EBMtotal_loss += lossE
            GENtotal_loss += lossG

            prof.step()
            if batch_idx >= 5:
                break
        
        tqdm_bar.set_description(f"Epoch {epoch}: EBM-Loss: {EBMtotal_loss / (BATCH_SIZE):.4f} GEN-Loss: {GENtotal_loss / (BATCH_SIZE):.4f}")

        if (epoch % SAMPLE_BREAK == 0 or epoch == NUM_EPOCHS):
            generated_data = generate_sample(Sampler, GENnet, EBMnet).reshape(-1, 1, 28, 28)
            img_grid = torchvision.utils.make_grid(generated_data, normalize=True)

            writer.add_image("Generated Samples -- Vanilla Pang Model", img_grid, global_step=epoch)

writer.close()

save_final_sample(generated_data, hyperparams=[NUM_EPOCHS, p0_SIGMA, GENERATOR_SIGMA])


plot_hist(Sampler, EBMnet, GENnet, x)
Sampler.batch_size = 500
X = train_dataset.data[:500].to(device).view(-1, data_dim)
plot_pdf(Sampler, EBMnet, GENnet, X.float(), data_dim)


