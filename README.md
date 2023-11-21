# 2023 IIB Project -- Exploring Variance and Gradients in Latent Space Energy-Based Models

* Author: [Prithvi Raj](pr478@cam.ac.uk)
* Supervisors: Mr. Justin Bunker, Prof. Mark Girolami
* Affiliation: University of Cambridge, Computational Statistics & Machine Learning Group
 
## Introduction

This repository is dedicated as a platform for my Master's thesis on deep generative models, as part of Cambridge University Engineering Despartment's (CUED) Integrated MEng Tripos course. It explores the impact of likelihood variance on the performance of a [Latent Space Energy-Based Prior Model]([url](https://arxiv.org/abs/2006.08205)) proposed by Pang et al. The study investigates whether improving the model's capacity to generate low-variance samples enhances overall performance. Incorporating thermodynamic integration, the experiment involves training both the original and an altered model, tracking gradients of log-likelihoods during training, and assessing generative capabilities through FID scores. The results, including variance against training iteration and FID score analysis, will provide insights into the nuanced relationship between likelihood variance, gradients, and model performance in latent space energy-based models.

* For more information regarding the exact nature of my research proposal, please refer to [my project brief](https://github.com/PritRaj1/IIB-Project-LatentEBM-Variance-Study/blob/main/DOCUMENTATION/Project_Proposal.pdf). 

* For information regarding my day-to-day progress, please refer to [my logbook](https://github.com/PritRaj1/IIB-Project-LatentEBM-Variance-Study/blob/main/DOCUMENTATION/LOGBOOK.ipynb), which is being kept as a requirement for the CUED IIB project.

* The theoretical concepts leveraged by this repository have been supported with reference material within the documentation provided. The code has been written and modularised by me, (Prithvi), as is demonstrated by the painstaking, statistical troubleshooting process laid out in the [logbook]([url](https://github.com/PritRaj1/IIB-Project-LatentEBM-Variance-Study/blob/main/DOCUMENTATION/LOGBOOK.ipynb)https://github.com/PritRaj1/IIB-Project-LatentEBM-Variance-Study/blob/main/DOCUMENTATION/LOGBOOK.ipynb).

## Breakdown of files
