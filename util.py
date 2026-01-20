#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: util.py
Description: Utility functions for multi-sequence MRI harmonization.

Author: Mengqi Wu
Email: mengqiw@unc.edu
Date: 01/12/2026

Reference:
    This code accompanies the manuscript titled:
    "Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style" (Under Review)
    	
	Please cite the preprint if you use this code: 
	https://doi.org/10.48550/arXiv.2601.08193

License: MIT License (see LICENSE file for details)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance


def get_mean_std(input, eps=1e-6):
	B, C = input.shape[:2]
	mean = torch.mean(input.view(B,C,-1),dim=2).view(B,C,1,1,1) # mean shape (B, C, 1, 1)
	std = torch.sqrt(torch.var(input.view(B,C,-1), dim=2)+eps).view(B, C, 1, 1,1)
	
	return mean, std

def AdaIN(content, style):
	assert content.shape[:2] == style.shape[:2]
	c_mean, c_std = get_mean_std(content)
	s_mean, s_std = get_mean_std(style)

	
	normalized = s_std*((content-c_mean)/c_std) + s_mean
	
	return normalized

def IN(content):
	c_mean, c_std = get_mean_std(content)

	
	normalized = (content-c_mean)/c_std
	return normalized

def Style_loss(input, target):
	mean_loss, std_loss = 0, 0

	for input_layer, target_layer in zip(input, target): 
		mean_input_layer, std_input_layer = get_mean_std(input_layer)
		mean_target_layer, std_target_layer = get_mean_std(target_layer)

		mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
		std_loss += F.mse_loss(std_input_layer, std_target_layer)

	return mean_loss+std_loss



def torch_gradmap(img):
    """
    Calculates the gradient magnitude (L2 norm) of a 5D tensor
    with a known dimension order of (B, C, Width, Height, Depth).
    """
    # Gradient along Width (x-axis, dimension 2)
    dw = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    # Gradient along Height (y-axis, dimension 3)
    dh = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    # Gradient along Depth (z-axis, dimension 4)
    dz = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

    # Align gradient components to a common grid.
    # The slicing logic remains the same as your version.
    dw_aligned = dw[:, :, :, :-1, :-1]
    dh_aligned = dh[:, :, :-1, :, :-1]
    dz_aligned = dz[:, :, :-1, :-1, :]

    # Calculate the L2 norm (magnitude) of the gradient vector
    grad_magnitude = torch.sqrt(dw_aligned**2 + dh_aligned**2 + dz_aligned**2)

    # Pad the result to restore the original image dimensions
    grad_map = F.pad(grad_magnitude, (1,0,1,0,1,0), "constant", 0)

    # 2. Apply logarithmic scaling to make the map denser
    # log1p(x) is equivalent to log(1+x), which handles zero values gracefully.
    grad_map = torch.log1p(grad_map)

    
    return grad_map

def torch_gradmap_aligned_L1(img):
    """
    Calculates a spatially aligned L1 Norm of the gradient.
    This provides a dense, robust structural map.
    Input shape: (B, C, H, W, D).
    """
    # Step 1: Calculate gradients
    dH = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    dW = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dD = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

    # Step 2: Align all components to a common grid
    dH_aligned = dH[:, :, :, :-1, :-1]
    dW_aligned = dW[:, :, :-1, :, :-1]
    dD_aligned = dD[:, :, :-1, :-1, :]

    # Step 3: Combine using the L1 Norm (sum of absolute values)
    # This is the key change.
    grad_l1_magnitude = torch.abs(dH_aligned) + torch.abs(dW_aligned) + torch.abs(dD_aligned)

    # Step 4: Pad to restore original size
    grad_map = F.pad(grad_l1_magnitude, (1, 0, 1, 0, 1, 0), "constant", 0)

    return grad_map

def torch_gradmap_average(img):
    """
    Calculates a spatially aligned average of gradients for a 5D tensor.

    This version correctly aligns the gradient components to a common grid
    before averaging them.

    Args:
        img (torch.Tensor): Input tensor with shape (B, C, H, W, D).

    Returns:
        torch.Tensor: The averaged gradient map with the same shape as the input.
    """
    # Step 1: Calculate the gradient along each spatial axis
    # ---------------------------------------------------------
    # Gradient along Height (H, axis 2)
    dH = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    # Gradient along Width (W, axis 3)
    dW = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    # Gradient along Depth (D, axis 4)
    dD = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

    # Step 2: Align all components to a common spatial grid
    # ---------------------------------------------------------
    # This is the crucial step that ensures spatial coherence.
    dH_aligned = dH[:, :, :, :-1, :-1]
    dW_aligned = dW[:, :, :-1, :, :-1]
    dD_aligned = dD[:, :, :-1, :-1, :]

    # Step 3: Combine the aligned components using a simple average
    # ---------------------------------------------------------
    grad_avg = (dH_aligned + dW_aligned + dD_aligned) / 3.0

    # Step 4: Pad the final map to restore the original image size
    # ---------------------------------------------------------
    # Pads the beginning of each spatial dimension (H, W, D) with one zero.
    grad_map = F.pad(grad_avg, (1, 0, 1, 0, 1, 0), "constant", 0)

    return grad_map

def norm_gradmap_percnetile(gradmap):
    # gradmap: shape [B, 1, H, W, D]
    if gradmap.min() < 0:
         flat = gradmap.abs().flatten(start_dim=2)  # [B, C, N]
    else:
        flat = gradmap.flatten(start_dim=2)  # [B, C, N]
    scale = torch.quantile(flat, 0.99, dim=2, keepdim=True) + 1e-8  # [B, C, 1]
    while scale.dim() < gradmap.dim():
        scale = scale.unsqueeze(-1)
    gradmap_norm = gradmap / scale  # Use gradmap_norm as condition
    gradmap_norm = gradmap_norm.clamp(-5.0, 5.0)

    return gradmap_norm

def grad_loss(grad_map1, grad_map2, loss_type):
	if loss_type == 'l1':
		return F.l1_loss(grad_map1, grad_map2)
	elif loss_type == 'l2':
		return F.mse_loss(grad_map1, grad_map2)
	else:
		raise ValueError("Wrong loss type, either l1 or l2")

def _downsample_3d(x, s):
    if s == 1:
        return x
    # Avg pool keeps scale-consistent low-frequency content
    return F.avg_pool3d(x, kernel_size=s, stride=s, padding=0)

def multiscale_grad_loss(img_pred, img_ref, scales=(1,2,4,8), loss_type='l1', percentile_norm=True):
    """
    Multi-scale gradient structural loss.
    img_pred, img_ref: (B,1,H,W,D)
    scales: pooling factors
    loss_type: 'l1' or 'l2'
    percentile_norm: apply your percentile normalization per scale (helps robustness)
    """
    total = 0.0
    count = 0
    for s in scales:
        p_ds = _downsample_3d(img_pred, s)
        r_ds = _downsample_3d(img_ref,  s)

        g_p = torch_gradmap_average(p_ds)
        g_r = torch_gradmap_average(r_ds)

        if percentile_norm:
            g_p = norm_gradmap_percnetile(g_p)
            g_r = norm_gradmap_percnetile(g_r)

        if loss_type == 'l1':
            l = F.l1_loss(g_p, g_r)
        elif loss_type == 'l2':
            l = F.mse_loss(g_p, g_r)
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        total += l
        count += 1
    return total / count


# Define the MLP-based style discriminator
class Style_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Style_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(input.size(0), -1))
    

class Style_Discriminator_3d(nn.Module):
	def __init__(self):
		super(Style_Discriminator_3d, self).__init__()
		self.main = nn.Sequential(
			nn.Conv3d(6, 64, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv3d(64, 128, 4, 2, 1, bias=False),
			nn.InstanceNorm3d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv3d(128, 256, 4, 2, 1, bias=False),
			nn.InstanceNorm3d(256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Flatten(),
			nn.Linear(256*6*6*2, 1)
		)

	def forward(self, input):
		return self.main(input)


def calculate_intensity_correlation(image1, image2):
    # Ensure the images are numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Calculate the mean of the images
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    # Subtract the mean from the images
    image1 -= mean1
    image2 -= mean2

    # Calculate the product of the images
    product = image1 * image2

    # Calculate the sum of the product
    sum_product = np.sum(product)

    # Calculate the sum of the squares of the images
    sum_square1 = np.sum(image1 ** 2)
    sum_square2 = np.sum(image2 ** 2)

    # Calculate the Pearson correlation coefficient
    correlation = sum_product / np.sqrt(sum_square1 * sum_square2)

    return correlation.mean()

def calculate_batch_intensity_correlation(batch1, batch2):
    batch_size = batch1.shape[0]
    correlations = []
    for i in range(batch_size):
        image1 = batch1[i, 0]  # Extract the 3D image from the batch
        image2 = batch2[i, 0]  # Extract the 3D image from the batch
        correlation = calculate_intensity_correlation(image1, image2)
        correlations.append(correlation)
    return np.mean(correlations)

def mean_wasserstein_distance(list1, list2):
    # Flatten the images and compute the Wasserstein distance for each pair
    distances = [wasserstein_distance(img1.flatten(), img2.flatten()) for img1, img2 in zip(list1, list2)]
    
    # Return the mean distance
    return np.mean(distances)

def calculate_batch_wasserstein_distance(batch1, batch2):
    batch_size = batch1.shape[0]
    distances = []
    for i in range(batch_size):
        image1 = batch1[i, 0]  # Extract the 3D image from the batch
        image2 = batch2[i, 0]  # Extract the 3D image from the batch
        distance = mean_wasserstein_distance([image1], [image2])
        distances.append(distance)
    return np.mean(distances)

def soft_histogram(x, bins=64, value_range=(0., 1.), σ=0.01, ignore=10):
    """
    x:     FloatTensor of shape [N_voxels] or [B, 1, W, H, D], values in [min, max]
    bins:  number of histogram bins
    value_range: (min_val, max_val) over which to compute bins
    σ:     kernel bandwidth (controls smoothness)
    """
    # Flatten x to [N, 1]
    flat = x.view(-1, 1)  # shape [N, 1]

    # Create bin centers on the same device as x
    min_val, max_val = value_range
    centers = torch.linspace(min_val, max_val, bins, device=x.device)  # [bins]
    centers = centers.view(1, bins)   # [1, bins]

    # Compute (flat - centers)/σ → [N, bins], then Gaussian weight
    diff = (flat - centers) / σ
    w = torch.exp(-0.5 * (diff ** 2))  # [N, bins]

    # Sum across voxels → unnormalized soft‐histogram [bins]
    hist = w.sum(dim=0)  # [bins]

    # Normalize so sum = 1
    hist = hist / (hist.sum() + 1e-8)
    return hist[ignore:], centers[:,ignore:]  # still a GPU tensor; differentiable wrt x

def wasserstein_1d_from_cdf(cdf1, cdf2):
    # cdf1, cdf2: FloatTensors of shape [bins], both on same device
    return torch.mean(torch.abs(cdf1 - cdf2))


def soft_hist_differentiable_wd(x_pred, x_ref, bins=64, value_range=(0., 1.), σ=0.01):
    # 1) Compute soft histograms
    
    hist1 = soft_histogram(x_pred, bins=bins, value_range=value_range, σ=σ)  # [bins]
    hist2 = soft_histogram(x_ref,  bins=bins, value_range=value_range, σ=σ)  # [bins]

    # 2) Compute cumulative sums → CDFs
    cdf1 = torch.cumsum(hist1, dim=0)  # [bins]
    cdf2 = torch.cumsum(hist2, dim=0)  # [bins]

    # 3) Return mean absolute difference
    return torch.mean(torch.abs(cdf1 - cdf2))

def differentiable_wd(hist1, hist2):

    # 2) Compute cumulative sums → CDFs
    cdf1 = torch.cumsum(hist1, dim=0)  # [bins]
    cdf2 = torch.cumsum(hist2, dim=0)  # [bins]

    # 3) Return mean absolute difference
    return torch.mean(torch.abs(cdf1 - cdf2))

def differentiable_wd_buffer(style_buffer):
    """
    style_buffer: list of [bins] tensors (histograms or CDFs), all differentiable
    Returns: mean WD over all unique pairs (as a scalar tensor)
    """
    n = len(style_buffer)
    if n < 2:
        return torch.tensor(0.0, device=style_buffer[0].device, dtype=style_buffer[0].dtype)
    wd_sum = 0.0
    count = 0
    # Compute CDFs if not already CDFs
    cdfs = [torch.cumsum(hist, dim=0) for hist in style_buffer]
    for i in range(n):
        for j in range(i + 1, n):
            wd_sum += torch.mean(torch.abs(cdfs[i] - cdfs[j]))
            count += 1
    return wd_sum / count

def soft_argmax(hist, centers, tau=0.0001):
    if not isinstance(hist, torch.Tensor):
        hist = torch.tensor(hist, dtype=torch.float32)
    w = F.softmax(hist / tau, dim=-1)
    return (w * centers).sum(dim=-1), w