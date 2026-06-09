#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: train_Stage2.py
Description: training script for the MMH Stage II: Target-Specific Fine Harmonization

Author: Mengqi Wu
Email: mengqiw@unc.edu
Date: 01/12/2026

Reference:
    This code accompanies the manuscript titled:
    "Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style" (Under Review)
		
	Please cite the preprint if you use this code: 
	https://doi.org/10.48550/arXiv.2601.08193

License: Apache-2.0 License (see LICENSE file for details)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import shutil
import sys
from pathlib import Path
import datetime
import argparse
import data.MRIdata as MRI
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.optim as optim
import torchvision
from PIL import Image
import random
import re
import math
from monai import transforms
from monai.data import DataLoader
from monai.networks.layers import Act
from monai.utils import set_determinism
from monai.losses.ssim_loss import SSIMLoss
from tqdm import tqdm
from itertools import cycle
from monai.inferers import DiffusionInferer
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import  DiffusionModelUNet, PatchDiscriminator
from monai.networks.schedulers import  DDIMScheduler
from open_clip import create_model_from_pretrained, get_tokenizer
from monai.data import MetaTensor
import util
from util import *
import pandas as pd
import torchvision.transforms as transforms
to_pil = transforms.ToPILImage()
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn

from monai import transforms

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_determinism(seed=seed)


def norm_mri(x, mask=None):
	# x: (B, 1, D, H, W)
	if mask is not None:
		x = x * mask
		mean = x.sum(dim=[2,3,4], keepdim=True) / (mask.sum(dim=[2,3,4], keepdim=True) + 1e-8)
		std = torch.sqrt(((x - mean) ** 2 * mask).sum(dim=[2,3,4], keepdim=True) / (mask.sum(dim=[2,3,4], keepdim=True) + 1e-8))
	else:
		mean = x.mean(dim=[2,3,4], keepdim=True)
		std = x.std(dim=[2,3,4], keepdim=True)
	return (x - mean) / (std + 1e-8)

# --- Add a small helper for per-step logging ---
def log_step(writer, tag_base, step, data: dict):
	"""
	data: {name: scalar}
	"""
	for k,v in data.items():
		writer.add_scalar(f"{tag_base}/{k}", v, step)

def visualize_batch_4D_tensor(img_volume, fn, save_pt, prefix=''):
	#### set normalize to Falase to reflect the true pixel values of MRI slices
	os.makedirs(save_pt, exist_ok=True)

	if image_min == -1.0 and img_volume.min() < 0:
		img_volume = (img_volume + 1.0) / 2.0  # [-1,1] -> [0,1]

	img_volume = torch.clamp(img_volume,0,1)

	grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=False,scale_each=True) # axial middle slices
	grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_a = (grid_a * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'a')
	save_path = save_pt / filename
	Image.fromarray(grid_a).save(save_path)


	grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=False,scale_each=True) # coronal middle slice
	grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_c = (grid_c * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'c')
	save_path = save_pt / filename
	Image.fromarray(grid_c).save(save_path)

	grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=False,scale_each=True) # saggital middle slice
	grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_s = (grid_s * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'s')
	save_path = save_pt / filename
	Image.fromarray(grid_s).save(save_path)

_pos_cache = {}
def get_sinusoidal_embeddings(n_position, embed_dim, device=None):
	key = (n_position, embed_dim)
	if key in _pos_cache:
		return _pos_cache[key].to(device) if device else _pos_cache[key]
	pe = torch.zeros(n_position, embed_dim)
	position = torch.arange(0, n_position, dtype=torch.float32).unsqueeze(1)
	div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	pe = pe.unsqueeze(0)  # (1, S, D)
	_pos_cache[key] = pe
	return pe.to(device) if device else pe

class SliceAggregator(nn.Module):
	"""
	Learns to pool slice embeddings from multiple views using attention.
	"""
	def __init__(self, embed_dim=512, num_heads=8):
		super().__init__()
		self.embed_dim = embed_dim
		self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
		self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
		self.norm = nn.LayerNorm(embed_dim)
		self.view_embeddings = nn.Embedding(3, embed_dim) # 0:axial, 1:coronal, 2:sagittal
		# MLP to fuse the three view summaries
		self.fuse_mlp = nn.Sequential(
			nn.Linear(embed_dim * 3, embed_dim,bias=True),
			nn.GELU(),
			nn.LayerNorm(embed_dim)
		)
		self.alpha = nn.Parameter(torch.tensor(0.1))  # residual scale

	def pool(self, embeds): # embeds: (B, S, D)
		B = embeds.shape[0]
		q = self.query.expand(B, -1, -1)
		out, _ = self.mha(q, embeds, embeds)
		out = out.squeeze(1)
		out = self.norm(out)
		return out

	def forward(self, emb_axial, emb_coronal, emb_sagittal):
		pooled = []
		for emb in (emb_axial, emb_coronal, emb_sagittal):
			if emb is None:
				# Use a zero tensor if a view is missing for some reason
				pooled.append(torch.zeros(emb_axial.shape[0], self.embed_dim, device=emb_axial.device))
			else:
				pooled.append(self.pool(emb))

		cat = torch.cat(pooled, dim=1)
		fused_raw = self.fuse_mlp(cat)
		mean_pooled = torch.stack(pooled, dim=0).mean(dim=0)
		fused = mean_pooled + self.alpha * fused_raw  # residual fusion
		return fused

class DF_CLIP_Finetune(Dataset):
	def __init__(self,style_rm_img_dir:str,noisy_latent_dir:str=None,annotation_file=None): 
		# assert Path(latent_dir).is_dir(), f'{latent_dir} is not a valid directory'
		# if annotation_file is not None: assert Path(annotation_file).is_file(), f'{annotation_file} is not a valid file'
		
		self.style_rm_img_dir = style_rm_img_dir
		if noisy_latent_dir:
			self.latent_dir = noisy_latent_dir
		else:
			self.latent_dir = None
		self.img_labels = annotation_file
		self.length = len(self.img_labels)
		self.transform = transforms.Compose([
				transforms.CenterSpatialCrop(roi_size=(144, 184, 184)),
				])

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		style_rm_img_pt = Path(self.style_rm_img_dir)
		fn  = str(self.img_labels.iloc[idx,0])
		
		found_files = list(style_rm_img_pt.glob(f'*{fn}.*'))
		
		if not found_files:
			raise FileNotFoundError(
				f"CRITICAL ERROR in DataLoader: Could not find any style-removed image for filename pattern '*{fn}.*' "
				f"in directory '{style_rm_img_pt}'. Please check the file paths and names."
			)
		style_rm_img_pt_full = str(found_files[0])
		site = self.img_labels.iloc[idx,1]
		if '.pt' in style_rm_img_pt_full:
			style_rm_img = torch.load(style_rm_img_pt_full,weights_only=True).float()
		if '.npy' in style_rm_img_pt_full:
			style_rm_img = torch.from_numpy(np.load(style_rm_img_pt_full)).float()
		if style_rm_img is None:
			raise ValueError(f'Failed to load style-removed image from {style_rm_img_pt_full}')
		if self.latent_dir:
			img_path_base = Path(self.latent_dir)
		
			latent_files = list(img_path_base.glob(f'*{fn}.pt'))
			if not latent_files:
				raise FileNotFoundError(
					f"CRITICAL ERROR in DataLoader: Could not find latent file for pattern '*{fn}.pt' "
					f"in directory '{img_path_base}'. Please check the file paths and names."
				)
			img_path_full = str(latent_files[0])			
			latent_volume = torch.load(img_path_full,weights_only=False)['latent'].float()
			condition = torch.load(img_path_full,weights_only=False)['condition'].float()
			class_emb = torch.load(img_path_full,weights_only=False)['class_emb']
			if len(latent_volume.shape) != 4:
				latent_volume = latent_volume.unsqueeze(0)
		

		style_rm_img = self.transform(style_rm_img)
		if isinstance(style_rm_img, MetaTensor):
			style_rm_img = style_rm_img.as_tensor()


		if self.latent_dir:
			example = {'latent':latent_volume,'condition':condition,'style_rm_img':style_rm_img,'fn':fn, 'class_emb':class_emb}
		else:
			example = {'style_rm_img':style_rm_img,'fn':fn}

		return example



def fdp(model, scheduler_ddim,input_img, class_emb, conditions,condition_mode='concat',ema_mean=None,ema_std=None):
	scheduler_ddim.set_timesteps(num_inference_steps=num_inference_fdp)
	img_noisy = inferer.reverse_sample(
		input_noise=input_img, diffusion_model=model, scheduler=scheduler_ddim,
		conditioning=conditions,mode=condition_mode, verbose=False,class_label=class_emb,
		ema_mean=ema_mean, ema_std=ema_std
	)

	return img_noisy

def rdp(model, scheduler_ddim,latent_noisy, class_emb, conditions,condition_mode='concat',ema_mean=None,ema_std=None):
	scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
	recon_images = inferer.sample(
		input_noise=latent_noisy, diffusion_model=model, scheduler=scheduler_ddim,
		conditioning=conditions,mode=condition_mode,verbose=False, class_label=class_emb,
		ema_mean=ema_mean, ema_std=ema_std
	)
	return recon_images


def style_removal(loader, desc = '', save_sample = False):
	with torch.inference_mode():
		# fdp -> rdp -> save style-removed latent
		progress_bar = tqdm(enumerate(loader),total=len(loader), ncols=150,leave=True,dynamic_ncols=True, mininterval=0.5)
		progress_bar.set_description(f"Style removal: {desc}")
		count = 0
		tqdm.write(f'Style removal conditioned on {condition_on}')


		for (step, batch) in progress_bar:
			batch_length = len(batch["image"])
			fn_list = batch["fn"]
			class_emb = batch['class_emb'].to(device)
			if class_emb is None:
				ema_mean_batch = None
				ema_std_batch = None
			else:
				# class_emb may be a tensor of shape (B,) or (1,)
				try:
					cls_list = [int(x) for x in class_emb.view(-1).cpu().tolist()]
				except Exception:
					cls_list = [int(class_emb.item())]
				ema_mean_batch = torch.tensor([EMA_mean[c] for c in cls_list], device=device, dtype=torch.float32)
				ema_std_batch = torch.tensor([EMA_std[c] for c in cls_list], device=device, dtype=torch.float32)


			image = batch['image'].to(device)

			if condition_on == 'grad':
				conditions = batch['image'].to(device)
				conditions = util.torch_gradmap_average(conditions)
				conditions = util.norm_gradmap_percnetile(conditions)
				conditions = torch.tanh(conditions.clamp(-10.0, 10.0))
				conditions = conditions * 0.5


			else:
				conditions = None

			if not (conditions is None):
				condition_mode = 'concat'
			else:
				condition_mode = 'crossattn'

			with autocast('cuda'):
				img_noisy = fdp(unet, scheduler_ddim,image, class_emb, conditions, condition_mode, ema_mean=ema_mean_batch, ema_std=ema_std_batch)
				recon_images = rdp(unet,scheduler_ddim,img_noisy, class_emb, conditions, condition_mode, ema_mean=ema_mean_batch, ema_std=ema_std_batch)

			for idx, recon_img in enumerate(recon_images):
				if image_min == -1.0:
					recon_img = torch.clamp(recon_img,min=-1.0,max=1.0)
				elif image_min == 0.0:
					recon_img = torch.clamp(recon_img,min=0.0,max=1.0)

				filename = fn_list[idx]
				save_pt = save_dir/'1_style_removed_images' / f'{filename}.npy'

				os.makedirs(save_pt.parent, exist_ok=True)

				np.save(save_pt, recon_img.detach().cpu().float().numpy()) ######### style removed image saved in orgianl range [0,1] or [-1,1], specified by image_min

				count += 1

			if save_sample:
				visualize_batch_4D_tensor(recon_images.detach().cpu().float(), 
							  '_'.join([f for f in fn_list]).replace('THP000',''), save_pt.parent/'sample_style_removed')
				visualize_batch_4D_tensor(batch['image'].detach().cpu().float(), 
							  '_'.join([f for f in fn_list]).replace('THP000',''), save_pt.parent/'sample_org')
				# visualize_batch_4D_tensor(img_noisy.detach().cpu().float(), fn_list, save_pt.parent/'sample_noisy')

			progress_bar.refresh()
			sys.stdout.flush()
				
		tqdm.write(f"Saved {count} style-removed latent representations to {save_dir}")

def match_mean_std(vol, ref, eps=1e-9, image_min=None, mask=None, clamp=True):
	"""
	Optional mask + optional clamp. Use clamp=False for perceptual/grad if wanting full gradient range.
	"""
	if mask is not None:
		m = (mask > 0)
		vol_mean = vol[m].mean()
		vol_std = vol[m].std()
		ref_mean = ref[m].mean()
		ref_std = ref[m].std()
	else:
		vol_mean = vol.mean()
		vol_std = vol.std()
		ref_mean = ref.mean()
		ref_std = ref.std()
	vol = (vol - vol_mean) / (vol_std + eps)
	out = vol * (ref_std + eps) + ref_mean
	if clamp:
		if image_min is not None:
			rmin = -1.0 if image_min == -1.0 else 0.0
		else:
			rmin = float(ref.min())
		out = torch.clamp(out, rmin, 1.0)
	return out

def scale_images(data_dir, target='CCF'):
	"""scale image to target mean and std"""
	print(f'{"#"*20} Applying mean and std scaling to images in {data_dir} {"#"*20}')
	count = 0
	
	for img_path in tqdm(Path(data_dir).glob('*.npy'), leave=False):
		img = np.load(img_path)
		
		if 'T2' in img_path.name:
			ref_files = sorted(Path(data_dir).glob(f'*{target}_T2.npy'))
			assert len(ref_files) > 0, f"No reference file found for target {target}_T2.npy in {data_dir}"
			ref = np.load(ref_files[0])
		else:
			ref_files = sorted(Path(data_dir).glob(f'*{target}.npy'))
			assert len(ref_files) > 0, f"No reference file found for target {target}.npy in {data_dir}"
			ref = np.load(ref_files[0])
			
		img_scaled = match_mean_std(torch.from_numpy(img), torch.from_numpy(ref),image_min=image_min)
		np.save(img_path, img_scaled.numpy())
		count +=1
	print(f'Scaled {count} images in {data_dir} to match {target} mean and std.')

def create_CLIP(which_clip='BiomedCLIP'):
	tqdm.write(f"Creating {which_clip} model...")
	if which_clip == 'BiomedCLIP':
		model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
		tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
	elif which_clip == 'OpenAI':
		model, preprocess = create_model_from_pretrained('hf-hub:timm/vit_base_patch16_clip_224.openai')
		tokenizer = get_tokenizer('hf-hub:timm/vit_base_patch16_clip_224.openai')
	model.to(device)

	return model, preprocess, tokenizer

def preprocess_with_grad(image):
	"""
	Preprocess the input image tensor while preserving gradient flow.
	
	Input:  (batch, 1, 184, 184)
	Output: (batch, 3, 224, 224)
	"""
	batch_size = image.shape[0]  # Preserve batch dimension

	# Ensure image is (batch, 1, H, W)
	assert image.ndim == 4 and image.shape[1] == 1, "Input should be (batch, 1, H, W)"

	# Map to [0,1] if inputs are in [-1,1]
	if 'image_min' in globals() and image_min == -1.0:
		image = (image + 1.0) / 2.0

	image = image.clamp(0.0, 1.0)

	# Calculate padding
	pad_h = (224 - image.shape[2]) // 2
	pad_w = (224 - image.shape[3]) // 2
	padding = (pad_w, pad_w, pad_h, pad_h)  # (left, right, top, bottom)
	# Apply zero padding
	image = F.pad(image, padding, mode='constant', value=0)  # (batch, 1, 224, 224)

	# Convert grayscale (1 channel) to RGB (3 channels) by repeating
	image = image.repeat(1, 3, 1, 1)  # (batch, 3, 224, 224)


	# Normalize using PyTorch tensors (Retains Gradient)
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(1, 3, 1, 1)
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(1, 3, 1, 1)
	image = (image - mean) / std  # Normalize while keeping gradient

	return image

def get_CLIP_embeddings(model, image, text=None):
	with autocast('cuda'):
		# print("before preprocessing require grad:", image.requires_grad)  # Should be True

		image = preprocess_with_grad(image)
		image = image.to(device).half()
		# print(f'after processing: {image.shape}')
		# print("after preprocessing require grad:", image.requires_grad)  # Should be True
		# print(image.shape)
		if text:
			tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
			context_length = 256 if which_clip == 'BiomedCLIP' else 77
			texts = tokenizer(text, context_length=context_length).to(device)

			img_embedding, text_embedding, logit_scale = model(image, texts)

			return img_embedding, text_embedding
		else:
			img_embedding = model.encode_image(image)
			return img_embedding


def get_volume_CLIP_embeddings(model, volume_3d, ignore_slices=40, max_slices=24, batch_size=16,aggregator=None):
	"""
	New function for multi-view attention pooling over 2D CLIP slice embeddings.
	volume_3d: tensor with shape (B, 1, D, H, W).
	volume_3d shape: (B, 1, X, Y, Z)
	Axis semantics:
	  dim 2 (X, left-right)    -> sweep for sagittal sequence (produces sagittal planes Y–Z)
	  dim 3 (Y, anterior-post) -> after permute becomes depth for coronal (planes X–Z)
	  dim 4 (Z, inferior-sup.) -> after permute becomes depth for axial (planes X–Y)
	"""
	global _slice_aggregator
	device = volume_3d.device
	embed_dim = 512 # BiomedCLIP/OpenAI default

	if isinstance(volume_3d, MetaTensor):
		volume_3d = volume_3d.as_tensor()

	# Use provided aggregator or global
	if aggregator is not None:
		agg = aggregator
	else:
		if _slice_aggregator is None:
			_slice_aggregator = SliceAggregator(embed_dim=embed_dim).to(device)
		agg = _slice_aggregator

	B, _, D, H, W = volume_3d.shape
	assert D > 2 * ignore_slices + 2, f"ignore_slices {ignore_slices} too large for depth {D}"

	def _embed_view(permutation, view_idx):
		# permutation is a tuple for permute
		vol_perm = volume_3d.permute(*permutation)  # (B,1,Depth,_,_)
		depth_axis = 2
		depth = vol_perm.shape[depth_axis]
		# Safe indices
		raw_idx = torch.linspace(ignore_slices, depth - 1 - ignore_slices, steps=max_slices, device=device)
		idx = torch.unique(raw_idx.round().long().clamp(0, depth - 1))
		# Gather slices
		slices = vol_perm[:, :, idx, :, :]  # (B,1,S,H,W)
		B_, _, S, H_, W_ = slices.shape
		flat = slices.permute(0, 2, 1, 3, 4).reshape(B_ * S, 1, H_, W_)  # (B*S,1,H,W)

		emb_chunks = []
		with autocast('cuda'):
			for start in range(0, flat.shape[0], batch_size):
				chunk = flat[start:start+batch_size]
				emb = get_CLIP_embeddings(model, chunk)  # (bs, D)
				emb_chunks.append(emb.float())
		emb = torch.cat(emb_chunks, dim=0).view(B_, S, -1)  # (B,S,D)

		pos = get_sinusoidal_embeddings(S, emb.shape[-1], device=device)  # (1,S,D)
		view_emb = agg.view_embeddings.weight[view_idx].view(1, 1, -1)  # (1,1,D)
		emb = emb + pos + view_emb

		emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize
		return emb

	# sagittal uses original (B,1,D,H,W)
	emb_sag = _embed_view((0,1,2,3,4), 2)
	# coronal: (B,1,H,D,W)
	emb_cor = _embed_view((0,1,3,2,4), 1)
	# axial: (B,1,W,D,H)
	emb_ax = _embed_view((0,1,4,2,3), 0)

	fused = agg(emb_ax, emb_cor, emb_sag)  # (B, D)
	return fused

def CLIP_SD_loss(vol_trans,
				 vol_src_content,
				 vol_tar_content,
				 vol_tar,
				 model,
				 w_dir=1.0,
				 w_mag=1.0,
				 detach_refs=True):
	"""
	Style displacement loss with directional (cosine) + magnitude components.

	vol_trans: translated/source-after-style volume (has gradients)
	vol_src_content, vol_tar_content, vol_tar: reference volumes (detached if detach_refs)

	Displacement vectors:
	  d_trans = f(vol_trans) - f(vol_src_content)
	  d_tar   = f(vol_tar)   - f(vol_tar_content)

	Losses:
	  Direction: 1 - cosine( normalize(d_trans), normalize(d_tar) )
	  Magnitude: | ||d_trans||_2 - ||d_tar||_2 |  (style strength alignment)

	Detaching reference paths blocks the aggregator from collapsing all embeddings jointly.
	Returns:
	total_loss, dir_loss, mag_loss, dir_cos, trans_disp_norm_mean, tar_disp_norm_mean

	"""
	ref_ctx = torch.inference_mode() if detach_refs else torch.enable_grad()
	# with torch.no_grad():
	vol_trans = match_mean_std(vol_trans, vol_tar.detach(), image_min=image_min, clamp=False)

	with ref_ctx:
		shadow_aggregator = SliceAggregator(embed_dim=512).to(vol_trans.device)
		shadow_aggregator.load_state_dict(_slice_aggregator.state_dict())
		shadow_aggregator.eval()
		tar_content_embed = get_volume_CLIP_embeddings(model, vol_tar_content,
			ignore_slices=ignore_slices, aggregator=shadow_aggregator,max_slices=max_CLIP_slices)
		tar_embed = get_volume_CLIP_embeddings(model, vol_tar,
			ignore_slices=ignore_slices, aggregator=shadow_aggregator,max_slices=max_CLIP_slices)

		src_content_embed = get_volume_CLIP_embeddings(model, vol_src_content,
		ignore_slices=ignore_slices, aggregator=shadow_aggregator,max_slices=max_CLIP_slices)

	src_trans_embed = get_volume_CLIP_embeddings(model, vol_trans,
		ignore_slices=ignore_slices, aggregator=_slice_aggregator,max_slices=max_CLIP_slices)

	raw_trans = src_trans_embed - src_content_embed.detach()
	raw_tar   = tar_embed - tar_content_embed

	norm_trans = F.normalize(raw_trans, dim=-1)
	norm_tar   = F.normalize(raw_tar, dim=-1)
	cos_sim = F.cosine_similarity(norm_trans, norm_tar).mean()
	dir_loss = 1 - cos_sim

	mag_loss = (raw_trans.norm(dim=-1) - raw_tar.norm(dim=-1)).abs().mean()

	total = w_dir * dir_loss + w_mag * mag_loss
	return (total,
		dir_loss.item(),
		mag_loss.item(),
		cos_sim.item(),
		raw_trans.norm(dim=-1).mean().item(),
		raw_tar.norm(dim=-1).mean().item())

def pre_compute_latent(T_trans,loader, desc = '', save_sample = False):
	'''
	Pre-compute the style-removed latent representations using FDP and save them.
	loader: DataLoader that returns style-removed images and their fn
	Style-removed image --> FDP --> noisy latent
	'''
	with torch.inference_mode():
		# fdp -> rdp -> save style-removed latent
		progress_bar = tqdm(enumerate(loader),total=len(loader), ncols=150,leave=True,mininterval=0.1)
		progress_bar.set_description(f"Pre-compute Latent: {desc}")
		count = 0

		for (step, batch) in progress_bar:
			batch_length = len(batch["image"])
			fn_list = batch["fn"]

			class_emb = batch['class_emb'].to(device)
			if class_emb is None:
				ema_mean_batch = None
				ema_std_batch = None
			else:
				# class_emb may be a tensor of shape (B,) or (1,)
				try:
					cls_list = [int(x) for x in class_emb.view(-1).cpu().tolist()]
				except Exception:
					cls_list = [int(class_emb.item())]
				ema_mean_batch = torch.tensor([EMA_mean[c] for c in cls_list], device=device, dtype=torch.float32)
				ema_std_batch = torch.tensor([EMA_std[c] for c in cls_list], device=device, dtype=torch.float32)
			images = batch["image"].to(device)

			tqdm.write(f'Pre-compute Latent conditioned on {condition_on}')

			if condition_on == 'grad':
				conditions = images.detach().clone()
				conditions = util.torch_gradmap_average(conditions)
				conditions = util.norm_gradmap_percnetile(conditions)
				conditions = torch.tanh(conditions.clamp(-10.0, 10.0))
				conditions = conditions * 0.5

			else:
				conditions = None


			if not (conditions is None):
				condition_mode = 'concat'
			else:
				condition_mode = 'crossattn'

			latent_noisy = fdp(unet,scheduler_ddim,images, class_emb, conditions,condition_mode, ema_mean=ema_mean_batch, ema_std=ema_std_batch)

			for idx, recon_latent in enumerate(latent_noisy):
				filename = fn_list[idx]
				save_pt = save_dir/'2_fdp_latents' / f'{filename}.pt'

				os.makedirs(save_pt.parent, exist_ok=True)

				torch.save({'latent': recon_latent.detach().cpu().float(),
							'condition':conditions[idx].detach().cpu().float(), #### save the sclaed condition
							'class_emb':class_emb[idx].detach().cpu()}, save_pt)  # save the noisy latent after FDP, without decoding and / scale_factor
				count += 1

			if save_sample:
				visualize_batch_4D_tensor(latent_noisy.detach().cpu().float(),
							   '_'.join([f for f in fn_list]).replace('THP000',''), save_pt.parent/'sample_noisy')
				visualize_batch_4D_tensor(batch['image'].detach().cpu().float(),
							   '_'.join([f for f in fn_list]).replace('THP000',''), save_pt.parent/'sample_org')

			progress_bar.refresh()
			sys.stdout.flush()

				
		tqdm.write(f"Saved {count} pre-computed latent representations to {save_dir}")


def style_recon_DDIM(model, optimizer, noisy_latent_loader, iterations=50, save_sample=False, epoch=0):
	scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)

	for iteration in tqdm(range(iterations), total=iterations, desc='Style recon iter', dynamic_ncols=True, leave=True, position=1):
		epoch_sr_loss = 0
		epoch_discriminator_loss = 0
		epoch_generator_loss = 0
		total_updates = 0

		for idx, batch in tqdm(enumerate(noisy_latent_loader), total=len(noisy_latent_loader),
							   dynamic_ncols=True, leave=False, position=2):
			conditioning = batch['condition'].to(device)
			class_emb = batch['class_emb'].to(device)
			modality = int(class_emb.item())
			if class_emb is None:
				ema_mean_batch = ema_std_batch = None
			else:
				cls_list = [int(x) for x in class_emb.view(-1).cpu().tolist()]
				ema_mean_batch = torch.tensor([EMA_mean[c] for c in cls_list], device=device)
				ema_std_batch = torch.tensor([EMA_std[c] for c in cls_list], device=device)

			latent = batch['latent'].to(device)
			fn_list = batch['fn']
			mode = 'concat' if conditioning is not None else 'crossattn'

			style_rm_tar_fn = fn_list[0]
			target_style_image_path = next(data_pt.glob(f'*{style_rm_tar_fn}.npy'))
			target_style_image = transform(torch.from_numpy(np.load(target_style_image_path))).unsqueeze(0).float().to(device)
			if isinstance(target_style_image, MetaTensor):
				target_style_image = target_style_image.as_tensor().to(device)


			if epoch > EMA_freeze_epochs:
				mean_tar = target_style_image[brain_mask > 0].mean()
				std_tar = target_style_image[brain_mask > 0].std()
				EMA_mean[modality] = EMA_mean[modality] * EMA_alpha + mean_tar.detach() * (1 - EMA_alpha)
				EMA_std[modality] = EMA_std[modality] * EMA_alpha + std_tar.detach() * (1 - EMA_alpha)

			timesteps_list = list(scheduler_ddim.timesteps)

			
			if SR_MODE == 'last_k':
				K = min(FINAL_K_SR, len(timesteps_list))
				# Fast-forward (no grad) through early steps
				with torch.inference_mode():
					for t in timesteps_list[:-K]:
						if mode == "concat":
							model_input = torch.cat([latent, conditioning], dim=1)
							model_output = model(
								model_input,
								timesteps=torch.tensor((t,), device=latent.device),
								context=None, class_labels=class_emb,
								ema_mean=ema_mean_batch, ema_std=ema_std_batch
							)
						else:
							model_output = model(
								latent,
								timesteps=torch.tensor((t,), device=latent.device),
								context=conditioning, class_labels=class_emb,
								ema_mean=ema_mean_batch, ema_std=ema_std_batch
							)
						latent, _ = scheduler_ddim.step(model_output, t, latent)
						latent = latent.detach()

				# Independent per-step updates on last K
				for t in timesteps_list[-K:]:
					latent_leaf = latent.clone().detach().requires_grad_(True)
					with autocast('cuda', enabled=True):
						if mode == "concat":
							model_input = torch.cat([latent_leaf, conditioning], dim=1)
							model_output = model(
								model_input,
								timesteps=torch.tensor((t,), device=latent_leaf.device),
								context=None, class_labels=class_emb,
								ema_mean=ema_mean_batch, ema_std=ema_std_batch
							)
						else:
							model_output = model(
								latent_leaf,
								timesteps=torch.tensor((t,), device=latent_leaf.device),
								context=conditioning, class_labels=class_emb,
								ema_mean=ema_mean_batch, ema_std=ema_std_batch
							)
						latent_next, pred_org = scheduler_ddim.step(model_output, t, latent_leaf)

						sr_loss = style_recon_loss(pred_org, target_style_image)
						loss_preceptual = perceptual_loss(
							pred_org * brain_mask, target_style_image * brain_mask
						) * perceptual_internal_scale

						loss_grad = util.multiscale_grad_loss(
							target_style_image * brain_mask, pred_org.float() * brain_mask,
							scales=(1, 2, 4, 8), loss_type=grad_loss_type
						)
						norm_pred_org = match_mean_std(pred_org, target_style_image, image_min=image_min)

						if weight_adv > 0:
							logits_fake = discriminator(norm_pred_org.float())[-1]
							generator_loss = adv_loss(
								logits_fake, target_is_real=True, for_discriminator=False
							)
						else:
							generator_loss = torch.tensor(0.0, device=latent.device)

						total_loss = (
							weight_sr * sr_loss
							+ weight_adv * generator_loss
							+ weight_preceptual * loss_preceptual
							+ weight_grad * loss_grad
						)

					optimizer.zero_grad(set_to_none=True)
					scaler.scale(total_loss).backward()
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					scaler.step(optimizer)
					scaler.update()

					if weight_adv > 0:
						with autocast('cuda', enabled=True):
							norm_pred_det = match_mean_std(
								pred_org.detach(), target_style_image.detach(), image_min=image_min
							)
							logits_fake = discriminator(norm_pred_det.float())[-1]
							loss_d_fake = adv_loss(
								logits_fake, target_is_real=False, for_discriminator=True
							)
							norm_target_image = match_mean_std(
								target_style_image.detach(), target_style_image.detach(), image_min=image_min
							)
							logits_real = discriminator(norm_target_image.float())[-1]
							loss_d_real = adv_loss(
								logits_real, target_is_real=True, for_discriminator=True
							)
							discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)
							loss_d = weight_adv * discriminator_loss
						optimizer_d.zero_grad(set_to_none=True)
						scaler_d.scale(loss_d).backward()
						scaler_d.unscale_(optimizer_d)
						torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
						scaler_d.step(optimizer_d)
						scaler_d.update()
						epoch_discriminator_loss += discriminator_loss.item()

					epoch_sr_loss += sr_loss.item()
					epoch_generator_loss += generator_loss.item()
					total_updates += 1
					latent = latent_next.detach()


			else:
				raise ValueError(f"Unknown SR_MODE: {SR_MODE}")

		# Epoch-level logging
		epoch_sr_loss /= max(1, total_updates)
		epoch_generator_loss /= max(1, total_updates)
		epoch_discriminator_loss /= max(1, total_updates)

		writer.add_scalar('Style_recon_loss', epoch_sr_loss, epoch * iterations + iteration)
		writer.add_scalar('Style_recon_G_loss', epoch_generator_loss, epoch * iterations + iteration)
		writer.add_scalar('Style_recon_D_loss', epoch_discriminator_loss, epoch * iterations + iteration)
		### Temp above
		writer.add_scalar('MMH_step2/Style_recon_loss', epoch_sr_loss, epoch * iterations + iteration)
		writer.add_scalar('MMH_step2/Style_recon_G_loss', epoch_generator_loss, epoch * iterations + iteration)
		writer.add_scalar('MMH_step2/Style_recon_D_loss', epoch_discriminator_loss, epoch * iterations + iteration)

		if save_sample and (iteration % 10 == 0):
			visualize_batch_4D_tensor(
				latent.detach().cpu().float(),
				'_'.join([f for f in fn_list]).replace('THP000',''),
				save_dir / '3_style_recon_images' / f'{epoch}-{iteration}',
				prefix='tar_recon_'
			)
			visualize_batch_4D_tensor(
				pred_org.detach().cpu().float(),
				'_'.join([f for f in fn_list]).replace('THP000',''),
				save_dir / '3_style_recon_images' / f'{epoch}-{iteration}',
				prefix='tar_pred_'
			)
			visualize_batch_4D_tensor(
				target_style_image.detach().cpu().float(),
				'_'.join([f for f in fn_list]).replace('THP000',''),
				save_dir / '3_style_recon_images' / f'{epoch}-{iteration}',
				prefix='tar_'
			)
	return model
			



def CLIP_style_DDIM(model, optimizer, optimizer_agg, n_content_latent_loader:DataLoader, train_target_loader, epoch = 0, save_sample = False):
	### input: fdp_latent from style-removed source iamge, style-removed target image, original target image
	global_step_base = epoch * 100000
	if isinstance(train_target_loader,list):
		t1_loader = cycle(train_target_loader[0])
		t2_loader = cycle(train_target_loader[1])
		multi_modality = True
	else:
		tar_loader = cycle(train_target_loader)
		multi_modality = False	

	scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)

	epoch_sd_loss = 0
	epoch_dir_loss = 0
	epoch_mag_loss = 0
	epoch_preceptual_loss = 0
	epoch_grad_loss = 0
	epoch_generator_loss = 0
	epoch_discriminator_loss = 0
	epoch_wd_loss = 0


	total_updates = 0
	for idx, batch in tqdm(enumerate(n_content_latent_loader),
									 total=len(n_content_latent_loader), dynamic_ncols=True,leave=False,position=1,desc='Style Transfer Batch'):
		conditioning = batch['condition'].to(device)
		class_label = batch['class_emb'].to(device).to(device)
		if class_label is None:
				ema_mean_batch = None
				ema_std_batch = None
		else:
			try:
				cls_list = [int(x) for x in class_label.view(-1).cpu().tolist()]
			except Exception:
				cls_list = [int(class_label.item())]
			ema_mean_batch = torch.tensor([EMA_mean[c] for c in cls_list], device=device, dtype=torch.float32)
			ema_std_batch = torch.tensor([EMA_std[c] for c in cls_list], device=device, dtype=torch.float32)
		latent = batch['latent'].to(device)
		fn_list = batch['fn']
		if not (conditioning is None):
			mode = 'concat'
		else:
			mode = 'crossattn'	

		if multi_modality:
			if class_label[0] == 0:
				tar_batch = next(t1_loader)
			else:
				tar_batch = next(t2_loader)
		else:
			tar_batch = next(tar_loader)	

		style_rm_target = tar_batch['style_rm_img'].to(device)
		style_rm_tar_fn = tar_batch['fn']
		target_image_path = next(data_pt.glob(f'*{style_rm_tar_fn[0]}.npy'))
		target_image = transform(torch.from_numpy(np.load(target_image_path))).unsqueeze(0).float().to(device) # original target img serve as style
		if isinstance(target_image, MetaTensor):
			target_image = target_image.as_tensor().to(device)
	
		progress_bar = tqdm(iter(scheduler_ddim.timesteps),total=len(scheduler_ddim.timesteps), 
					  dynamic_ncols=True,leave=False, position=2,desc='Timestep:')
		timesteps_list = list(scheduler_ddim.timesteps)

				
		if ST_MODE == "last_k":
			K = min(FINAL_K_ST, len(timesteps_list))
			# Fast-forward without gradients
			with torch.inference_mode():
				for t in timesteps_list[:-K]:
					if mode == "concat":
						model_input = torch.cat([latent, conditioning], dim=1)
						model_output = model(
							model_input,
							timesteps=torch.tensor((t,), device=latent.device),
							context=None, class_labels=class_label,
							ema_mean=ema_mean_batch, ema_std=ema_std_batch
						)
					else:
						model_output = model(
							latent,
							timesteps=torch.tensor((t,), device=latent.device),
							context=conditioning, class_labels=class_label,
							ema_mean=ema_mean_batch, ema_std=ema_std_batch
						)
					latent, _ = scheduler_ddim.step(model_output, t, latent)
					latent = latent.detach()

			# Independent updates on last K
			for t in timesteps_list[-K:]:
				latent_leaf = latent.clone().detach().requires_grad_(True)
				with autocast('cuda', enabled=True):
					if mode == "concat":
						model_input = torch.cat([latent_leaf, conditioning], dim=1)
						model_output = model(
							model_input,
							timesteps=torch.tensor((t,), device=latent_leaf.device),
							context=None, class_labels=class_label,
							ema_mean=ema_mean_batch, ema_std=ema_std_batch
						)
					else:
						model_output = model(
							latent_leaf,
							timesteps=torch.tensor((t,), device=latent_leaf.device),
							context=conditioning, class_labels=class_label,
							ema_mean=ema_mean_batch, ema_std=ema_std_batch
						)
					latent_next, pred_org = scheduler_ddim.step(model_output, t, latent_leaf)
					recon_image = pred_org

					recon_C = norm_mri(recon_image, brain_mask if use_mask else None)
					src_C   = norm_mri(batch['style_rm_img'].to(device), brain_mask if use_mask else None)


					loss_preceptual = perceptual_loss(
						recon_C * brain_mask, src_C * brain_mask
					) * perceptual_internal_scale

					loss_grad = util.multiscale_grad_loss(
                        src_C * brain_mask, 
                        recon_C * brain_mask,
                        scales=(1, 2, 4, 8), 
                        loss_type=grad_loss_type
                    )

					norm_recon_org = match_mean_std(pred_org, target_image.detach(), image_min=image_min)

					if weight_adv > 0:
						logits_fake = discriminator(norm_recon_org.float())[-1]
						generator_loss = adv_loss(
							logits_fake, target_is_real=True, for_discriminator=False
						)
					else:
						generator_loss = torch.tensor(0.0, device=latent.device)

					sd_loss, dir_loss, mag_loss, dir_cos, trans_norm, tar_norm = CLIP_SD_loss(
						recon_image, batch['style_rm_img'].to(device),
						style_rm_target, target_image, CLIP_model,
						w_dir=weight_dir, w_mag=weight_mag
					)
					total_loss = (
						weight_sd * sd_loss
						+ weight_preceptual * loss_preceptual
						+ weight_grad * loss_grad
						+ weight_adv * generator_loss
					)

				optimizer.zero_grad(set_to_none=True)
				optimizer_agg.zero_grad(set_to_none=True)
				scaler.scale(total_loss).backward()
				scaler.unscale_(optimizer)
				scaler.unscale_(optimizer_agg)
				unet_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
				# aggregator grad norm
				if any(p.requires_grad for p in _slice_aggregator.parameters()):
					agg_grad_norm = torch.nn.utils.clip_grad_norm_(_slice_aggregator.parameters(), 1.0)
				else:
					agg_grad_norm = torch.tensor(0.0, device=latent_leaf.device)
					
				scaler.step(optimizer)
				if epoch >= freeze_agg_epochs:
					scaler.step(optimizer_agg)
				scaler.update()

				if weight_adv > 0:
					with autocast('cuda', enabled=True):
						norm_recon_det = match_mean_std(
							recon_image.detach(), target_image.detach(), image_min=image_min
						)
						logits_fake = discriminator(norm_recon_det.float())[-1]
						loss_d_fake = adv_loss(
							logits_fake, target_is_real=False, for_discriminator=True
						)
						norm_target_image = match_mean_std(
							target_image.detach(), target_image.detach(), image_min=image_min
						)
						logits_real = discriminator(norm_target_image.float())[-1]
						loss_d_real = adv_loss(
							logits_real, target_is_real=True, for_discriminator=True
						)
						discriminator_loss = 0.5 * (loss_d_fake + loss_d_real)
						loss_d = weight_adv * discriminator_loss
					optimizer_d.zero_grad(set_to_none=True)
					scaler_d.scale(loss_d).backward()
					scaler_d.unscale_(optimizer_d)
					D_grad_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0).item()
					scaler_d.step(optimizer_d)
					scaler_d.update()
					epoch_discriminator_loss += discriminator_loss.item()
				else:
					D_grad_norm = torch.tensor(0.0, device=latent_leaf.device)
					discriminator_loss = torch.tensor(0.0, device=latent_leaf.device)

				with torch.inference_mode():
					recon_image_scaled = match_mean_std(
						recon_image.detach(), target_image.detach(), image_min=image_min
					)
					hist_pred, _ = soft_histogram(
						recon_image_scaled[brain_mask > 0],
						bins=100, value_range=(image_min, 1.0), σ=0.01, ignore=2
					)
					hist_tar, _ = soft_histogram(
						target_image[brain_mask > 0],
						bins=100, value_range=(image_min, 1.0), σ=0.01, ignore=2
					)
					train_WD = differentiable_wd(hist_pred, hist_tar)
					epoch_wd_loss += train_WD.item()

				epoch_sd_loss += sd_loss.item()
				epoch_dir_loss += dir_loss
				epoch_mag_loss += mag_loss
				epoch_preceptual_loss += loss_preceptual.item()
				epoch_grad_loss += loss_grad.item()
				epoch_generator_loss += generator_loss.item()
				total_updates += 1
				latent = latent_next.detach()
				progress_bar.set_postfix({'timestep': t,})
				step_id = global_step_base + total_updates

				if total_updates % 40 == 0:
					# Unweighted raw components
					log_step(writer, "step2_debug/raw_loss", step_id, {
						"sd": sd_loss.item(),
						"dir": dir_loss,
						"mag": mag_loss,
						"perc": loss_preceptual.item(),
						"grad": loss_grad.item(),
						"advG": generator_loss.item(),
						"advD": discriminator_loss.item()
					})
					# Weighted contributions
					log_step(writer, "step2_debug/weighted_loss", step_id, {
						"sd": (weight_sd * sd_loss).item(),
						"perc": (weight_preceptual * loss_preceptual).item(),
						"grad": (weight_grad * loss_grad).item(),
						"advG": (weight_adv * generator_loss).item()
					})
					# CLIP displacement diagnostics
					log_step(writer, "step2_debug/clip_displacement", step_id, {
						"dir_cos": dir_cos,
						"d_trans_norm": trans_norm,
						"d_tar_norm": tar_norm
					})
					# Intensity stats
					mask_idx = brain_mask > 0
					log_step(writer, "step2_debug/intensity", step_id, {
						"recon_mean": recon_image[mask_idx].mean().item(),
						"recon_std": recon_image[mask_idx].std().item(),
						"target_mean": target_image[mask_idx].mean().item(),
						"target_std": target_image[mask_idx].std().item()
					})
					# Grad norms + LRs
					log_step(writer, "step2_debug/grads", step_id, {
						"unet": unet_grad_norm,
						"aggregator": agg_grad_norm,
						"discriminator": D_grad_norm
					})

					unet_lr = next(pg['lr'] for pg in optimizer.param_groups if pg.get('name') == 'unet')
					agg_lr = next(pg['lr'] for pg in optimizer_agg.param_groups if pg.get('name') == 'aggregator_decay')
					agg_fuse_lr = next((pg['lr'] for pg in optimizer_agg.param_groups if pg.get('name') == 'aggregator_fuse'), 0.0)
					log_step(writer, "step2_debug/lr", step_id, {
						"unet": unet_lr,
						"aggregator": agg_lr,
						"aggregator_fuse": agg_fuse_lr,
						"D": optimizer_d.param_groups[0]['lr'],
					})

			
					with torch.no_grad():
						if not hasattr(_slice_aggregator, "_prev_fuse_W"):
							_slice_aggregator._prev_fuse_W = _slice_aggregator.fuse_mlp[0].weight.detach().clone()
						fuse_W = _slice_aggregator.fuse_mlp[0].weight.detach()
						fuse_W_delta = (fuse_W - _slice_aggregator._prev_fuse_W).norm().item()
						_slice_aggregator._prev_fuse_W = fuse_W.clone()
					fuse_W_grad = 0.0
					if _slice_aggregator.fuse_mlp[0].weight.grad is not None:
						fuse_W_grad = _slice_aggregator.fuse_mlp[0].weight.grad.detach().norm().item()


					log_step(writer, "step2_debug/aggregator", step_id, {
						"fuse_W_norm": fuse_W.norm().item(),
						"fuse_W_delta_norm": fuse_W_delta,
						"fuse_W_grad_norm": fuse_W_grad,
						"query_norm": _slice_aggregator.query.norm().item(),
					})

		if save_sample and 'recon_image_scaled' in locals():
			visualize_batch_4D_tensor(latent.detach().cpu().float(), 
								'_'.join([f.replace('THP000','') for f in fn_list]), save_dir/'4_style_translated_images' / f'{epoch}',prefix='Recon_src_')
			visualize_batch_4D_tensor(recon_image_scaled.detach().cpu().float(), 
								'_'.join([f.replace('THP000','') for f in fn_list]), save_dir/'4_style_translated_images' / f'{epoch}',prefix='Pred_src_')
			visualize_batch_4D_tensor(batch['style_rm_img'].float(),
								'_'.join([f.replace('THP000','') for f in fn_list]), save_dir/'4_style_translated_images' / f'{epoch}',prefix='src_rm_')
			visualize_batch_4D_tensor(style_rm_target.detach().cpu().float(),
								'_'.join([f.replace('THP000','') for f in style_rm_tar_fn]), save_dir/'4_style_translated_images' / f'{epoch}',prefix='tar_rm_')
			visualize_batch_4D_tensor(target_image.detach().cpu().float(),
								'_'.join([f.replace('THP000','') for f in style_rm_tar_fn]), save_dir/'4_style_translated_images' / f'{epoch}',prefix='tar')


	epoch_sd_loss /= total_updates
	epoch_dir_loss /= total_updates
	epoch_mag_loss /= total_updates
	epoch_preceptual_loss /= total_updates
	epoch_grad_loss /= total_updates
	epoch_generator_loss /= total_updates
	epoch_discriminator_loss /= total_updates
	epoch_wd_loss /= total_updates

	writer.add_scalar('MMH_step2/Style_transfer_style_loss', epoch_sd_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_dir_loss', epoch_dir_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_mag_loss', epoch_mag_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_preceptual_loss', epoch_preceptual_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_grad_loss', epoch_grad_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_generator_loss', epoch_generator_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_discriminator_loss', epoch_discriminator_loss, epoch)
	writer.add_scalar('MMH_step2/Style_transfer_WD_loss', epoch_wd_loss, epoch)
	
	
	# Memory cleanup
	if idx < (len(n_content_latent_loader) - 1):  
		del model_input, model_output, pred_org
	else:
		del model_input, model_output
	torch.cuda.empty_cache()
	
	if save_ckp:
		ckp_save_pt = save_dir/'model_ckp'
		os.makedirs(ckp_save_pt, exist_ok=True)
		torch.save({
			'model': model.state_dict(),
			'discriminator': discriminator.state_dict(),
			'aggregator': _slice_aggregator.state_dict(),
			'optimizer': optimizer.state_dict(),
			'optimizer_d': optimizer_d.state_dict(),
			'optimizer_agg': optimizer_agg.state_dict(),
			'scaler': scaler.state_dict(),
			'scaler_d': scaler_d.state_dict(),
			'lr_scheduler': lr_scheduler.state_dict(),
			'lr_scheduler_d': lr_scheduler_d.state_dict(),
			'lr_scheduler_agg': lr_scheduler_agg.state_dict(),
			'epoch': epoch,
			'epoch_total': epoch_total,
			'EMA_mean': EMA_mean,
			'EMA_std': EMA_std,
			'best_val_WD': best_val_WD,
			'torch_rng_state': torch.get_rng_state(),          # NEW
			'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
			'numpy_rng_state': np.random.get_state(),          # NEW
			'py_rng_state': random.getstate(),
		}, ckp_save_pt / f'intermediate.pt')
	
	return model

def CLIP_val(model,val_loader,epoch=0, save_sample = False):
	global best_val_WD
	model.eval()
	val_WD_total = 0
	count = 0
	with torch.inference_mode():
		for batch in tqdm(val_loader, total=len(val_loader), desc='Validation', ncols=150, leave=False, mininterval=0):
			sys.stdout.flush()
			conditioning = batch['condition'].to(device)
			class_label = batch['class_emb'].to(device).to(device)
			if class_label is None:
				ema_mean_batch = None
				ema_std_batch = None
			else:
				try:
					cls_list = [int(x) for x in class_label.view(-1).cpu().tolist()]
				except Exception:
					cls_list = [int(class_label.item())]
				ema_mean_batch = torch.tensor([EMA_mean[c] for c in cls_list], device=device, dtype=torch.float32)
				ema_std_batch = torch.tensor([EMA_std[c] for c in cls_list], device=device, dtype=torch.float32)


			if class_label.item() == 0:
				fn_insert=''
			if class_label.item() == 1:
				fn_insert='_T2'
			latent = batch['latent'].to(device)
			fn_list = batch['fn']
			subject = fn_list[0].split('_')[0]  
			if not (conditioning is None):
				mode = 'concat'
			else:
				mode = 'crossattn'
			target_fn =  f'{subject}_{target_site}{fn_insert}.npy'

			
			scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
			recon_images = rdp(model,scheduler_ddim,latent,class_label,conditioning,mode, ema_mean=ema_mean_batch, ema_std=ema_std_batch)
			recon_images=recon_images.detach()
		
			target_image_path = next(data_pt_val.glob(target_fn))
			target_image = transform(torch.from_numpy(np.load(target_image_path))).unsqueeze(0).float().to(device) # original target img serve as style
		
			if isinstance(target_image, MetaTensor):
				target_image = target_image.as_tensor().to(device)
	
			recon_images = match_mean_std(recon_images, target_image, image_min=image_min)

			hist_pred, centers = soft_histogram(recon_images[brain_mask>0], bins=100, value_range=(image_min,1.0), σ=0.01,ignore=2)
			hist_target, centers = soft_histogram(target_image[brain_mask>0], bins=100, value_range=(image_min,1.0), σ=0.01,ignore=2)
			val_WD = differentiable_wd(hist_pred, hist_target)
			val_WD_total += val_WD.item()
			count += 1


			if save_sample or val_only:
				for idx, recon_latent in enumerate(recon_images):
					filename = fn_list[idx]
					save_pt = save_dir/'5_val'/str(epoch) / f'{filename}'

					os.makedirs(save_pt.parent, exist_ok=True)
					if image_min == -1.0:
						recon_latent = (recon_latent + 1.0) / 2.0
					else:
						recon_latent = recon_latent.clamp(0.0, 1.0)
					np.save(save_pt, recon_latent.cpu().numpy())

		if not val_only:
			val_WD_avg = val_WD_total / count
			writer.add_scalar('Style_transfer_val_WD', val_WD_avg, epoch)
			writer.add_scalar('MMH_step2/Style_transfer_val_WD', val_WD_avg, epoch)
			if val_WD_avg < best_val_WD:
				best_val_WD = val_WD_avg
				if save_ckp:
					ckp_save_pt = save_dir/'model_ckp'
					os.makedirs(ckp_save_pt, exist_ok=True)
					torch.save({
						'model': model.state_dict(),
						'discriminator': discriminator.state_dict(),
						'aggregator': _slice_aggregator.state_dict(),
						'optimizer': optimizer.state_dict(),
						'optimizer_d': optimizer_d.state_dict(),
						'optimizer_agg': optimizer_agg.state_dict(),
						'scaler': scaler.state_dict(),
						'scaler_d': scaler_d.state_dict(),
						'lr_scheduler': lr_scheduler.state_dict(),
						'lr_scheduler_d': lr_scheduler_d.state_dict(),
						'lr_scheduler_agg': lr_scheduler_agg.state_dict(),
						'epoch': epoch,
						'epoch_total': epoch_total,
						'EMA_mean': EMA_mean,
						'EMA_std': EMA_std,
						'best_val_WD': best_val_WD,
						'torch_rng_state': torch.get_rng_state(),          # NEW
						'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
						'numpy_rng_state': np.random.get_state(),          # NEW
						'py_rng_state': random.getstate(),
					}, ckp_save_pt / f'{epoch}.pt')
			else:
				shutil.rmtree(save_pt.parent)


def test_loader():
	
	for batch in train_loader:
		print('train_loader')
		style_rm_img = batch['style_rm_img']
		condition = batch['condition']
		latent_volume = batch['latent']
		fn = batch['fn']
		print(fn)
		print(style_rm_img.shape)
		print(condition.shape)
		print(latent_volume.shape)

		break

	for batch in val_loader:
		print('val_loader')
		style_rm_img = batch['style_rm_img']
		condition = batch['condition']
		latent_volume = batch['latent']
		fn = batch['fn']
		print(fn)
		print(style_rm_img.shape)
		print(condition.shape)
		print(latent_volume.shape)

		break

def load_ckp(ckp_dir, which_load):
	model_ckp_files = [f for f in ckp_dir.glob('*.pt') if re.search(r'\d+\.pt$', f.name)]
	if model_ckp_files:
		if which_load == 'latest':
			latest_ckp = max(model_ckp_files, key=os.path.getctime)
			ckp = torch.load(latest_ckp)
			epoch_start = int(latest_ckp.stem)+1
			print(f'Latest checkpoint found: {latest_ckp}')
		elif isinstance(which_load, int):
		   # Load a specific checkpoint by epoch number
			specific_ckp = ckp_dir / f'{which_load}.pt'
			if specific_ckp.exists():
				ckp = torch.load(specific_ckp)
				epoch_start = which_load + 1
			else:
				raise FileNotFoundError(f"Checkpoint {specific_ckp} does not exist.")
		else:
			raise ValueError("which_load must be 'latest' or an integer epoch number.")
	
		print(f'Resuming from epoch: {epoch_start}')
	else:
		print(f'No checkpoint found in {ckp_dir}, starting from scratch.')
		ckp = None
		epoch_start = 0
	
	return ckp, epoch_start


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="MMH Stage II: Target-Specific Fine Harmonization")

	# Run parameters
	parser.add_argument('--run_name', type=str, default='DEFINE_YOUR_RUN_NAME_HERE', help='Run name')
	parser.add_argument('--epoch_total', type=int, default=30, help='Total epochs')
	parser.add_argument('--Resume', action='store_true', help='Resume from checkpoint')
	parser.add_argument('--val_only', action='store_true', help='Run validation only')
	parser.add_argument('--resume_from', type=str, default='PATH_TO_CKP_DIR', help='Path to checkpoint dir for resuming')
	parser.add_argument('--val_dir', type=str, default='PATH_TO_VAL_DIR', help='Path to validation directory')
	
	# Sampling / Inference parameters
	parser.add_argument('--num_train_ddim', type=int, default=50)
	parser.add_argument('--num_inference_fdp', type=int, default=35)
	parser.add_argument('--num_inference_rdp', type=int, default=25)

	# Training setup
	parser.add_argument('--bs', type=int, default=4)
	parser.add_argument('--image_min', type=float, default=-1.0)
	parser.add_argument('--disable_use_mask', action='store_true')
	parser.add_argument('--disable_save_sample', action='store_true')
	parser.add_argument('--disable_save_ckp', action='store_true')
	parser.add_argument('--grad_loss_type', type=str, default='l1')
	parser.add_argument('--norm', type=str, default='AdaIN')
	parser.add_argument('--condition_on', type=str, default='grad')

	# Weights and Schedule
	parser.add_argument('--weight_sr', type=float, default=1.0)
	parser.add_argument('--sd_floor', type=float, default=1.0)
	parser.add_argument('--weight_sd_max', type=float, default=5.0)
	parser.add_argument('--sd_ramp_epochs', type=int, default=8)
	parser.add_argument('--weight_dir', type=float, default=1.0)
	parser.add_argument('--weight_mag', type=float, default=1.5)
	parser.add_argument('--weight_preceptual', type=float, default=0.0)
	parser.add_argument('--perceptual_internal_scale', type=float, default=1.0)
	parser.add_argument('--weight_grad', type=float, default=0.5)
	parser.add_argument('--freeze_agg_epochs', type=int, default=0)
	parser.add_argument('--warmup_epochs', type=int, default=2)
	parser.add_argument('--adv_floor', type=float, default=0.0) # experimental, not used in current version
	parser.add_argument('--weight_adv_max', type=float, default=0.0)
	parser.add_argument('--adv_ramp_start_offset', type=int, default=0)
	parser.add_argument('--adv_ramp_duration', type=int, default=15)

	# CLIP
	parser.add_argument('--which_clip', type=str, default='BiomedCLIP')
	parser.add_argument('--ignore_slices', type=int, default=50)
	parser.add_argument('--max_CLIP_slices', type=int, default=24)

	# LRs
	parser.add_argument('--initial_lr', type=float, default=5e-7)
	parser.add_argument('--initial_lr_d', type=float, default=1e-7)
	parser.add_argument('--initial_slice_aggregator_lr', type=float, default=6e-7)

	# Paths & Data
	parser.add_argument('--Stage1_pt', type=str, default='PATH_TO_STAGE1_PRETRAINED_MODEL/stage1_model.pth')
	parser.add_argument('--data_pt', type=str, default='PATH_TO_TRAIN_DATA_DIR')
	parser.add_argument('--data_pt_val', type=str, default='PATH_TO_VAL_DATA_DIR')
	parser.add_argument('--target_site', type=str, default='UCI', help='Target site identifier used in filenames')
	parser.add_argument('--train_tsvs', nargs='+', default=['PATH_TO_TRAIN_T1.tsv', 'PATH_TO_TRAIN_T2.tsv'])
	parser.add_argument('--val_tsvs', nargs='+', default=['PATH_TO_VAL_T1.tsv', 'PATH_TO_VAL_T2.tsv'])
	parser.add_argument('--brain_mask', type=str, default='PATH_TO_BRAIN_MASK_FILE.npy')
	
	# Parse args
	args = parser.parse_args()

	# Unpack args
	num_train_ddim = args.num_train_ddim
	num_inference_fdp = args.num_inference_fdp
	num_inference_rdp = args.num_inference_rdp
	run_name = args.run_name
	epoch_total = args.epoch_total
	Resume = args.Resume
	val_only = args.val_only
	bs = args.bs
	image_min = args.image_min
	use_mask = not args.disable_use_mask
	save_sample = not args.disable_save_sample
	save_ckp = not args.disable_save_ckp
	grad_loss_type = args.grad_loss_type
	norm = args.norm
	condition_on = args.condition_on if args.condition_on.lower() != 'none' else None
	weight_sr = args.weight_sr
	sd_floor = args.sd_floor
	weight_sd_max = args.weight_sd_max
	sd_ramp_epochs = args.sd_ramp_epochs
	weight_dir = args.weight_dir
	weight_mag = args.weight_mag
	weight_preceptual = args.weight_preceptual
	perceptual_internal_scale = args.perceptual_internal_scale
	weight_grad = args.weight_grad
	freeze_agg_epochs = args.freeze_agg_epochs
	warmup_epochs = args.warmup_epochs
	adv_floor = args.adv_floor
	weight_adv_max = args.weight_adv_max
	adv_ramp_start = sd_ramp_epochs + args.adv_ramp_start_offset
	adv_ramp_end = adv_ramp_start + args.adv_ramp_duration
	which_clip = args.which_clip
	ignore_slices = args.ignore_slices
	max_CLIP_slices = args.max_CLIP_slices
	initial_lr = args.initial_lr
	initial_lr_d = args.initial_lr_d
	initial_slice_aggregator_lr = args.initial_slice_aggregator_lr
	Stage1_pt = args.Stage1_pt
	target_site = args.target_site

	FINAL_K_SR = 10          # style_recon_DDIM final-K window
	FINAL_K_ST = 10          # CLIP_style_DDIM final-K window
	SR_MODE = "last_k" # options: "per_t" (old), "last_k"
	ST_MODE = "last_k" # options: "per_t" (old), "last_k"
	EMA_freeze_epochs = -1
	EMA_alpha = 0.8

	if torch.cuda.is_available():
		device = torch.device("cuda:0") 
		print(f"Visible GPUs: {torch.cuda.device_count()}")
		print(f"Current GPU Index (PyTorch): {torch.cuda.current_device()}")
		print(f"GPU Name: {torch.cuda.get_device_name(0)}")
	else:
		device = torch.device("cpu")
		print("CUDA is not available.")
	##############################################

	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
	print(f'{now}_{run_name}')

	epoch_start = 0

	if val_only:
		resume_from = Path(args.val_dir)
		print(f'{"#"*20}Validation from: {resume_from} {"#"*20}')
		save_dir = resume_from.parent
		model_ckp_dir = save_dir / 'model_ckp'
		if model_ckp_dir.exists():
			ckp, epoch_start = load_ckp(model_ckp_dir, 32)  
	elif Resume:
		resume_from = Path(args.resume_from)
		print(f'Resume from: {resume_from}')
		save_dir = resume_from.parent
		writer=SummaryWriter(f'runs/{save_dir.name}')

		model_ckp_dir = save_dir / 'model_ckp'
		if model_ckp_dir.exists():
			ckp, epoch_start = load_ckp(model_ckp_dir, 'latest')  # Load the latest checkpoint
		else:
			print(f'No checkpoint found in {model_ckp_dir}, starting from scratch.')
			ckp = None
			epoch_start = 0
	else:
		if 'TEMP' not in run_name:
			writer=SummaryWriter(f'TBLOG/{now}_{run_name}')
		save_dir = Path(f'log/Stage2/{now}_{run_name}')

	if not save_dir.exists():
		os.makedirs(save_dir)
	elif not (Resume or val_only):
		assert len(os.listdir(save_dir))==0,'Log dir exist!'

	LDM_pt_file = torch.load(Stage1_pt)

	#################################### Define Dataloader ################################################################
	data_pt = Path(args.data_pt)
	data_pt_val = Path(args.data_pt_val)

	# combine source and target
	lb_train_src_tar_t1t2_combined = pd.concat([pd.read_csv(tsv_path, sep='\t') for tsv_path in args.train_tsvs], ignore_index=True)
	
	lb_val_src_tar_t1t2_combined = pd.concat([pd.read_csv(tsv_path, sep='\t') for tsv_path in args.val_tsvs], ignore_index=True)

		
	lb_train_tar = lb_train_src_tar_t1t2_combined[lb_train_src_tar_t1t2_combined['site'] == target_site]
	lb_train_tar_t1 = lb_train_tar[~lb_train_tar['filename'].str.endswith('T2')]
	lb_train_tar_t2 = lb_train_tar[lb_train_tar['filename'].str.endswith('T2')]

	lb_val_tar = lb_val_src_tar_t1t2_combined[lb_val_src_tar_t1t2_combined['site'] == target_site]
	lb_val_tar_t1 = lb_val_tar[~lb_val_tar['filename'].str.endswith('T2')]
	lb_val_tar_t2 = lb_val_tar[lb_val_tar['filename'].str.endswith('T2')]

	train_dataset = MRI.DWITHP_t1t2(data_pt, lb_train_src_tar_t1t2_combined, lb_train_tar_t1, lb_train_tar_t2, image_min=image_min)
	val_dataset = MRI.DWITHP_t1t2(data_pt_val, lb_val_src_tar_t1t2_combined, lb_val_tar_t1, lb_val_tar_t2, image_min=image_min)
	train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True,drop_last=False)
	val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True,drop_last=False)
	brain_mask = torch.from_numpy(np.load(args.brain_mask)).unsqueeze(0).float().to(device) # (1,1,144,184,184)





	############# end define dataloader ##################
	parameters = f"""
	run_name = '{run_name}',
	num_train_ddim = {num_train_ddim},
	num_inference_fdp = {num_inference_fdp},
	num_inference_rdp = {num_inference_rdp},
	use_mask = {use_mask},
	image_min = {image_min},
	weitht_sr = {weight_sr},
	sd_floor = {sd_floor},
	weight_sd_max = {weight_sd_max},
	sd_ramp_epochs = {sd_ramp_epochs},
	weight_dir = {weight_dir},
	weight_mag = {weight_mag},
	weight_preceptual = {weight_preceptual},
	perceptual_internal_scale = {perceptual_internal_scale},
	weight_grad = {weight_grad},
	EMA_alpha = {EMA_alpha},
	weight_adv_max = {weight_adv_max},
	adv_floor = {adv_floor},
	adv_ramp_start = {adv_ramp_start},
	adv_ramp_end = {adv_ramp_end},
	grad_loss_type = '{grad_loss_type}',
	freeze_agg_epochs = {freeze_agg_epochs},
	warmup_epochs = {warmup_epochs},
	initial_lr = {initial_lr},
	initial_lr_d = {initial_lr_d},
	initial_lr_slice_aggregator = {initial_slice_aggregator_lr},
	condition_on = {condition_on},
	norm = {norm},
	target_site = '{target_site}',
	which_clip = '{which_clip}',
	ignore_slices_in_CLIP_embedding = {ignore_slices},
	max_CLIP_slices = {max_CLIP_slices},

	EMA_freeze_epochs = {EMA_freeze_epochs},
	SR_MODE = '{SR_MODE}',
	FINAL_K_SR = {FINAL_K_SR},
	ST_MODE = '{ST_MODE}',
	FINAL_K_ST = {FINAL_K_ST},



	LDM_pt = {Stage1_pt}
	"""

  
	print(parameters)
	with open(save_dir / 'parameters.txt', 'w') as f:
		f.write(run_name+'\n')
		f.write(parameters)


	if True:
		try:
			script_path = Path(__file__).resolve()
			destination_path = save_dir / f"{script_path.stem}_{now}.py"
			shutil.copy2(script_path, destination_path)
			print(f"Saved script snapshot to {destination_path}")
		except NameError:
			print("Could not save script snapshot: `__file__` is not defined (e.g., in an interactive session).")
		except Exception as e:
			print(f"Error saving script snapshot: {e}")

	# exit()

	transform = transforms.Compose([
	transforms.CenterSpatialCrop(roi_size=(144, 184, 184)),
	transforms.ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=image_min, b_max=1.0, clip=True),
	])

	############# define model ##################
	# unet
	unet = DiffusionModelUNet(
		spatial_dims=3,
		in_channels=2,
		out_channels=1,
		num_res_blocks=2,
		channels=(32,64,256, 256),
		attention_levels=(False,False, True, True),
		num_head_channels=(0,0,32, 32),
		norm_num_groups=16,
		use_flash_attention=True,
		num_class_embeds=2,
		norm=norm
	)
	unet.to(device)

	unet.load_state_dict(LDM_pt_file['unet_state_dict'])
	tqdm.write('LDM weighted loaded!')
	EMA_mean = LDM_pt_file['ema_mean']
	EMA_std = LDM_pt_file['ema_std']
	print(f'LDM ema_mean: {EMA_mean}, ema_std: {EMA_std}')

	## Discriminator
	discriminator = PatchDiscriminator(
		spatial_dims=3,
		num_layers_d=4,
		channels=64,
		in_channels=1,
		out_channels=1,
		kernel_size=7,
		activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
		norm="INSTANCE", 
		bias=False,
		padding=3,
	)
	discriminator.to(device)

	perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet50_23datasets",is_fake_3d=False)
	perceptual_loss.to(device)



	scheduler_ddim = DDIMScheduler(num_train_timesteps=num_train_ddim, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195,clip_sample=False)

	inferer = DiffusionInferer(scheduler_ddim)

	scaler = GradScaler()
	scaler_d = GradScaler()

	# Initialize the aggregator here to add its parameters to the optimizer
	global _slice_aggregator
	_slice_aggregator = SliceAggregator(embed_dim=512).to(device)
	# initialize as simple mean _slice_aggregator...
	W = _slice_aggregator.fuse_mlp[0].weight   # shape [embed_dim, 3*embed_dim]
	b = _slice_aggregator.fuse_mlp[0].bias
	with torch.no_grad():
		W.zero_()
		embed_dim = W.shape[0]
		for i in range(embed_dim):
			W[i, i] = 1/3
			W[i, i + embed_dim] = 1/3
			W[i, i + 2*embed_dim] = 1/3
		b.fill_(0.05)
		# Make residual path contribute from the start
		_slice_aggregator.alpha.data.fill_(0.2)

	styleRM_data_pt = Path(f'{save_dir}/1_style_removed_images')
	noisy_latent_pt = Path(f'{save_dir}/2_fdp_latents')


	if not styleRM_data_pt.exists():
		unet.eval()
		print(f'Start style removal!')
		style_removal(train_loader, desc = 'Train', save_sample = save_sample)
		style_removal(val_loader, desc = 'Val', save_sample = save_sample)
		scale_images(styleRM_data_pt,target=target_site)

	else:
		print(f'Style removed images exist! Skip style removal!')
	######### style removed images saved in original range, specified by image_min #########



	########### Initialize dataloaders for pre-compute latents ##############
	train_dataset = MRI.DWITHP_t1t2(styleRM_data_pt, lb_train_src_tar_t1t2_combined, lb_train_tar_t1, lb_train_tar_t2)
	train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2,drop_last=False,persistent_workers=True,pin_memory=True)
	val_dataset = MRI.DWITHP_t1t2(styleRM_data_pt, lb_val_src_tar_t1t2_combined, lb_val_tar_t1, lb_val_tar_t2)
	val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2, drop_last=False,persistent_workers=True,pin_memory=True)

	
	if not noisy_latent_pt.exists():
		print(f'Start pre-compute latents!')
		pre_compute_latent(num_train_ddim,train_loader, desc = 'Train', save_sample = save_sample)
		pre_compute_latent(num_train_ddim,val_loader, desc = 'Val', save_sample = save_sample)
	else:
		print(f'Pre-compute latents exist! Skip pre-compute latents!')


	######################################## MMH Journal 2025 DWITHP ###################

	lb_train_src = lb_train_src_tar_t1t2_combined[~(lb_train_src_tar_t1t2_combined['site'] == target_site)]
	lb_train_tar = lb_train_src_tar_t1t2_combined[lb_train_src_tar_t1t2_combined['site'] == target_site]

	lb_train_tar_t1 = lb_train_tar[~lb_train_tar['filename'].str.endswith('T2')]
	lb_train_tar_t2 = lb_train_tar[lb_train_tar['filename'].str.endswith('T2')]

	bs_finetune = 1

	train_src_dataset = DF_CLIP_Finetune(styleRM_data_pt,noisy_latent_dir=noisy_latent_pt,annotation_file=lb_train_src)
	train_src_loader = DataLoader(
		train_src_dataset, batch_size=bs_finetune, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)
	train_tar_dataset = DF_CLIP_Finetune(styleRM_data_pt, noisy_latent_dir=noisy_latent_pt, annotation_file=lb_train_tar)
	train_tar_loader_t1t2 = DataLoader(
		train_tar_dataset, batch_size=bs_finetune, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)
	train_tar_dataset_t1 = DF_CLIP_Finetune(styleRM_data_pt, noisy_latent_dir=noisy_latent_pt, annotation_file=lb_train_tar_t1)
	train_tar_dataset_t2 = DF_CLIP_Finetune(styleRM_data_pt, noisy_latent_dir=noisy_latent_pt, annotation_file=lb_train_tar_t2)
	train_tar_loader_t1 = DataLoader(
		train_tar_dataset_t1, batch_size=bs_finetune, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)
	train_tar_loader_t2 = DataLoader(
		train_tar_dataset_t2, batch_size=bs_finetune, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)

	val_src_tar_t1t2_dataset = DF_CLIP_Finetune(styleRM_data_pt, noisy_latent_dir=noisy_latent_pt, annotation_file=lb_val_src_tar_t1t2_combined)
	val_src_tar_t1t2_loader = DataLoader(
		val_src_tar_t1t2_dataset, batch_size=bs_finetune, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)

	inference_src_tar_t1t2_dataset = DF_CLIP_Finetune(styleRM_data_pt, noisy_latent_dir=noisy_latent_pt, annotation_file=pd.concat([lb_train_src_tar_t1t2_combined,lb_val_src_tar_t1t2_combined]))
	inference_src_tar_t1t2_loader = DataLoader(
		inference_src_tar_t1t2_dataset, batch_size=bs_finetune, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=False
	)



	print(f"{'#'*20} Start Fine-tuning {'#'*20}")
	print(f'Train: number of source samples: {len(train_src_dataset)}',f'number of target samples: {len(train_tar_dataset)}')
	print(f'Val: number of val samples: {len(val_src_tar_t1t2_dataset)}')



	# ---- fine-grained param groups for the aggregator ----
	agg_decay, agg_no_decay, agg_fuse = [], [], []
	for n, p in _slice_aggregator.named_parameters():
		if not p.requires_grad:
			continue
		if n.startswith("fuse_mlp.0.") or n == "alpha":  # first Linear + residual gate
			agg_fuse.append(p)
		elif n.endswith(".bias") or ("norm" in n) or n.startswith("fuse_mlp.2."):  # LN/bias: no weight-decay
			agg_no_decay.append(p)
		else:
			agg_decay.append(p)

	optimizer = optim.Adam(
		[
			{'params': unet.parameters(), 'lr': initial_lr, 'betas': (0.9, 0.999), 'name': 'unet'},
		]
	)

	optimizer_agg = optim.Adam(
		[
			{'params': agg_decay, 'lr': initial_slice_aggregator_lr, 'weight_decay': 1e-4, 'betas': (0.9, 0.999), 'name': 'aggregator_decay'},
			{'params': agg_no_decay, 'lr': initial_slice_aggregator_lr, 'weight_decay': 0.0, 'betas': (0.9, 0.999), 'name': 'aggregator_no_decay'},
			{'params': agg_fuse, 'lr': max(3e-6, initial_slice_aggregator_lr * 2), 'weight_decay': 0.0, 'betas': (0.9, 0.999), 'name': 'aggregator_fuse'},
		]
	)


	# Sanity checks
	assert any(id(p) == id(_slice_aggregator.fuse_mlp[0].weight)
			for pg in optimizer_agg.param_groups for p in pg['params']), "Aggregator fuse_mlp weight not in optimizer_agg"
	assert any(pg.get('name') == 'unet' for pg in optimizer.param_groups), "Missing 'unet' param group name"
	assert not any(pg.get('name', '').startswith('aggregator') for pg in optimizer.param_groups), "Aggregator params should not be in main optimizer"



	optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=initial_lr_d,betas=(0.5, 0.99))
	adv_loss = PatchAdversarialLoss(criterion="least_squares")


	if (Resume or val_only) and ckp is not None:
		# ...existing code...
		prev_total = ckp.get('epoch_total', epoch_total)
	else:
		prev_total = epoch_total

	def lr_lambda(epoch):
		if epoch < warmup_epochs:
			return float(epoch + 1) / warmup_epochs
		# Use the original horizon up to its end; extend with the new total afterwards
		horizon = prev_total if epoch <= prev_total else epoch_total
		prog = (epoch - warmup_epochs) / max(1, (horizon - warmup_epochs))
		return 0.5 * (1 + math.cos(math.pi * prog))

	lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
	lr_scheduler_agg = LambdaLR(optimizer_agg, lr_lambda=lr_lambda)
	lr_scheduler_d = LambdaLR(optimizer_d, lr_lambda=lr_lambda)

	style_recon_loss = F.l1_loss

	unet.train()

	CLIP_model, preprocess, tokenizer = create_CLIP(which_clip=which_clip)
	CLIP_model.eval()
	for param in CLIP_model.parameters():
		param.requires_grad = False

	prev_total = epoch_total
	if (Resume or val_only) and ckp is not None:
		print(f'Loading Ckp from epoch {epoch_start-1}.pt')
		unet.load_state_dict(ckp['model'])
		discriminator.load_state_dict(ckp['discriminator'])
		_slice_aggregator.load_state_dict(ckp['aggregator'])
		optimizer.load_state_dict(ckp['optimizer'])
		optimizer_agg.load_state_dict(ckp['optimizer_agg'])
		optimizer_d.load_state_dict(ckp['optimizer_d'])
		scaler.load_state_dict(ckp['scaler'])
		scaler_d.load_state_dict(ckp['scaler_d'])
		lr_scheduler.load_state_dict(ckp['lr_scheduler'])
		if 'lr_scheduler_agg' in ckp:
			lr_scheduler_agg.load_state_dict(ckp['lr_scheduler_agg'])
		lr_scheduler_d.load_state_dict(ckp['lr_scheduler_d'])
		# Restore EMA stats if present
		if 'EMA_mean' in ckp and 'EMA_std' in ckp:
			EMA_mean = ckp['EMA_mean']
			EMA_std = ckp['EMA_std']
			print('Restored EMA_mean / EMA_std')
		if 'best_val_WD' in ckp:
			best_val_WD = ckp['best_val_WD']
			print(f'Restored best_val_WD={best_val_WD:.6f}')
		# RNG states (optional)
		try:
			if 'torch_rng_state' in ckp:
				torch.set_rng_state(ckp['torch_rng_state'])
			if torch.cuda.is_available() and 'cuda_rng_state_all' in ckp and ckp['cuda_rng_state_all'] is not None:
				torch.cuda.set_rng_state_all(ckp['cuda_rng_state_all'])
			if 'numpy_rng_state' in ckp:
				np.random.set_state(ckp['numpy_rng_state'])
			if 'py_rng_state' in ckp:
				random.setstate(ckp['py_rng_state'])
			print('RNG states restored')
		except Exception as e:
			print(f'RNG state restore skipped: {e}')
		print(f'All ckp loaded!')
	if val_only:
		print(f'Validation only mode! Start validation!')
		CLIP_val(unet,val_src_tar_t1t2_loader,epoch=epoch_start-1)
		sys.exit(0)


	best_val_WD = float('inf')
	for i in tqdm(range(epoch_start,epoch_total),desc='Global Step', ncols=150,leave=True, position=0,initial=epoch_start,total=epoch_total):
		if i < adv_ramp_start:
			weight_adv = adv_floor
		elif i >= adv_ramp_end:
			weight_adv = weight_adv_max
		else:
			frac_adv = (i - adv_ramp_start) / max(1, (adv_ramp_end - adv_ramp_start))
			weight_adv = adv_floor + (weight_adv_max - adv_floor) * frac_adv
		writer.add_scalar('train_util/weight_adv', weight_adv, i)

		# Style displacement weight ramp
		if i <= sd_ramp_epochs:
			frac_sd = i / max(1, sd_ramp_epochs)
			weight_sd = sd_floor + (weight_sd_max - sd_floor) * frac_sd
		else:
			weight_sd = weight_sd_max
		writer.add_scalar('train_util/weight_sd', weight_sd, i)
		# Freeze / unfreeze aggregator
		if i < freeze_agg_epochs:
			_slice_aggregator.eval()
			for p in _slice_aggregator.parameters():
				p.requires_grad = False
		else:
			_slice_aggregator.train()
			for p in _slice_aggregator.parameters():
				p.requires_grad = True

		print(f"[Epoch {i}] current_weight_adv={weight_adv:.4f} "
			f"unet_lr={optimizer.param_groups[0]['lr']:.2e} "
			f"agg_lr={optimizer_agg.param_groups[0]['lr']:.2e} "
			f"D_lr={optimizer_d.param_groups[0]['lr']:.2e}"
			f" weight_sd={weight_sd:.4f}"
			)

		unet.train()
			
		
		unet = style_recon_DDIM(unet, optimizer, train_tar_loader_t1t2, iterations=8, save_sample = save_sample, epoch=i) ### for SRPBS
		unet = CLIP_style_DDIM(unet, optimizer, optimizer_agg, train_src_loader, train_tar_loader_t1t2,epoch=i, save_sample = save_sample) ##### for SRPBS
		print(f'Validation at epoch {str(i)}')
		CLIP_val(unet,val_src_tar_t1t2_loader,i,save_sample = save_sample)

		lr_scheduler.step()
		lr_scheduler_d.step()
		lr_scheduler_agg.step()
	
	ckp_save_pt = save_dir/'model_ckp'
	os.makedirs(ckp_save_pt, exist_ok=True)
	torch.save({
		'model': unet.state_dict(),
		'discriminator': discriminator.state_dict(),
		'aggregator': _slice_aggregator.state_dict(),
		'optimizer': optimizer.state_dict(),
		'optimizer_d': optimizer_d.state_dict(),
		'optimizer_agg': optimizer_agg.state_dict(),
		'scaler': scaler.state_dict(),
		'scaler_d': scaler_d.state_dict(),
		'lr_scheduler': lr_scheduler.state_dict(),
		'lr_scheduler_d': lr_scheduler_d.state_dict(),
		'lr_scheduler_agg': lr_scheduler_agg.state_dict(),
		'epoch': epoch_total,
		'epoch_total': epoch_total,
		'EMA_mean': EMA_mean,               # NEW
		'EMA_std': EMA_std,                 # NEW
		'best_val_WD': best_val_WD,         # NEW
		'torch_rng_state': torch.get_rng_state(),          # NEW
		'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
		'numpy_rng_state': np.random.get_state(),          # NEW
		'py_rng_state': random.getstate(),
	}, ckp_save_pt / f'final_ckp.pt')