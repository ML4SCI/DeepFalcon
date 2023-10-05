from backward_diffusion import model
from forward_diffusion import forward_diffusion_sample, get_index_from_list, show_tensor_image
from forward_diffusion import sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, posterior_variance
from train import device
import torch
from torch import nn
import math


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    # print("before calling model")
    noise_pred = model(x_noisy, t)        #Feeding the forward diffused image to the model
    # print("after calling model")
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):                 # Call model to predict noise in the image and retur denoised image. Apply noise if not yet reached last step.
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (                                    # Call model (current image - noise prediction)
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, math.ceil((i/stepsize)) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()