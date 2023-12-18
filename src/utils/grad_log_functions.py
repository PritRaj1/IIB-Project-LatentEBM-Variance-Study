
import torch

def EBM_grad_log_fn(EBM, z):
        
        # Compute gradient of log p_a(x) w.r.t. z
        f_z = EBM.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        
        return grad_f_z - (z / (EBM.p0_sigma * EBM.p0_sigma)) # This is GRAD log[p_a(x)])

def vanillaGEN_grad_log_fn(GEN, z, x, EBM_model):
        
        # Compute gradient of log[p(x | z)] w.r.t z
        g_z = GEN.forward(z.view(z.size(0), -1, 1, 1))
        log_gz = -(torch.norm(x-g_z, dim=-1)**2) / (2.0 * GEN.lkhood_sigma * GEN.lkhood_sigma)
        grad_log_gz = torch.autograd.grad(log_gz.sum(), z, create_graph=True)[0]

        # Compute gradient of log[p(z)] w.r.t z
        f_z = EBM_model.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        grad_log_prior = grad_f_z - (z / (EBM_model.p0_sigma * EBM_model.p0_sigma))

        return grad_log_gz + grad_log_prior # This is GRAD log[ p(x | z) * p(z) ]

def temperedGEN_grad_log_fn(GEN, z, x, EBM_model):
        
        # Compute gradient of t * log[p(x | z)] w.r.t z
        g_z = GEN.forward(z)
        log_llood = - GEN.current_temp * (torch.norm(x-g_z, dim=-1)**2) / (2.0 * GEN.lkhood_sigma * GEN.lkhood_sigma)
        grad_log_llhood = torch.autograd.grad(log_llood.sum(), z, create_graph=True)[0]
        
        # Compute gradient of log[p(z)] w.r.t z
        f_z = EBM_model.forward(z)
        grad_f_z = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0] # Gradient of f_a(z)
        grad_log_prior = grad_f_z - (z / (EBM_model.p0_sigma * EBM_model.p0_sigma))
        
        return grad_log_llhood + grad_log_prior # This is GRAD log[ p(x | z)^t * p(z) ]