def EBM_loss(z_prior, z_posterior, EBMmodel):
    en_pos = EBMmodel(z_posterior.detach())
    en_neg = EBMmodel(z_prior.detach())
            
    return (en_pos - en_neg).mean()