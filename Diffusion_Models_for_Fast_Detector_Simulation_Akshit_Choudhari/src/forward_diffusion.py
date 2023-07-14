import numpy as np
import h5py
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import json


import torch
import torchvision
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from torchvision import transforms

def show_tensor_image(image):

    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),         #Scale from [-1,+1] to [0,1]
        # transforms.Normalize(mean=mean, std=std),
        # transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda im: np.array(im)),       # convert to NumPy array
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    # image = reverse_transforms(combined)
    image = reverse_transforms(image)

    combined = torch.sum(torch.from_numpy(image), dim=-1, keepdim=True)   # Combine Track, ECAL, and HCAL channels

    # print(image.shape)
    # print(type(image))
    # print(image)

    # plt.imshow(image, cmap='viridis', vmin=0.001, vmax=10, interpolation='nearest')
    # plt.imshow(combined, cmap='viridis', vmin=0.001, vmax=10, interpolation='nearest')


     # Set zero-valued cells as blank white
    combined[combined == 0] = np.nan
    # Plot the image
    plt.imshow(combined[:,:,0], cmap='viridis', vmin=-0.5, vmax=2.0, interpolation='nearest')


def linear_beta_schedule(timesteps, start=0.000001, end=0.0003):   #start and end need to be carefully chosen
    print("Using linear beta scheduler")
    return torch.linspace(start, end, timesteps)

def quadratic_beta_schedule(timesteps, start=0.0000000001, end=0.0000001):
    print("Using quadratic beta scheduler")
    return torch.linspace(start**2, end**2, timesteps).sqrt()

def exponential_beta_schedule(timesteps, start=0.000001, end=0.0003):
    print("Using exponential beta scheduler")
    return torch.exp(torch.linspace(torch.log(start), torch.log(end), timesteps))

def cosine_beta_schedule(timesteps, start=0.000001, end=0.0003):
    print("Using cosine beta scheduler")
    cos_anneal = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, timesteps)))
    return start + (end - start) * cos_anneal


def get_index_from_list(vals, t, x_shape):    # Returns a specific index t of a passed list of values vals while considering the batch dimension.
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):   #Take image and timestep and returns noisy version of the image
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
# betas = linear_beta_schedule(timesteps=T)
# betas = quadratic_beta_schedule(timesteps=T)
# betas = exponential_beta_schedule(timesteps=T)
betas = cosine_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


plt.figure(figsize=(55,55))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

# Simulate forward diffusion
image = next(iter(train_loader))[0]

# print(image.type)
show_tensor_image(image)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, math.ceil((idx/stepsize)) + 1)
    show_tensor_image(image)
    image, noise = forward_diffusion_sample(image, t)
    # show_tensor_image(image)