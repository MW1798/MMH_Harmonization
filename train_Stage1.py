#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: train_Stage1.py
Description: training script for the MMH Stage I: Sequence-Specific Global Harmonization

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

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import shutil
from pathlib import Path
import datetime
import MRIdata as MRI
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import torch.optim as optim
import torchvision
from PIL import Image
import pandas as pd
from util import soft_histogram, differentiable_wd, soft_argmax
from monai.data import DataLoader
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from torch.utils.tensorboard import SummaryWriter
import util

def center_crop(tensor, target_shape):
	_, _, d, h, w = tensor.shape
	td, th, tw = target_shape

	# Check if the tensor is already in the target shape
	if (d, h, w) == (td, th, tw):
		return tensor

	start_d = (d - td) // 2
	start_h = (h - th) // 2
	start_w = (w - tw) // 2
	return tensor[:, :, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

def get_style_weight(epoch, max_weight=1.0, rampup_epochs=100):
	if epoch < rampup_epochs:
		return max_weight * (epoch / rampup_epochs)
	else:
		return max_weight

def train():
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
		norm = norm
	)
	unet.to(device)

	
	scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195,clip_sample=True) 
	scheduler_ddim = DDIMScheduler(num_train_timesteps=num_train_ddim, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=clip_sample)

	# define infer
	inferer = DiffusionInferer(scheduler, style_stats=style_stats)
	optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_diff, 'min', patience=lr_patience, verbose=True) # 


	scaler = GradScaler()
	gs = 0

 
	if Resume:
		state_dict_ldm = torch.load(resume_from)
		unet.load_state_dict(state_dict_ldm['unet_state_dict'])
		optimizer_diff.load_state_dict(state_dict_ldm['optimizer_diff'])
		scaler.load_state_dict(state_dict_ldm['scaler_state_dict'])
		lr_scheduler.load_state_dict(state_dict_ldm['lr_scheduler_state_dict'])
		ema_mean = state_dict_ldm['ema_mean']
		ema_std = state_dict_ldm['ema_std']
		ema_hist = state_dict_ldm['ema_hist']
		best_val_loss_total = state_dict_ldm.get('best_val_loss_total', 1e9)
		best_val_style_loss_total = state_dict_ldm.get('best_val_style_loss_total', 1e9)
		epoch_start = state_dict_ldm.get('epoch', 0)
		tqdm.write(f'All DDPM state dict loaded!, starting from {epoch_start}')
	else:
		ema_hist, ema_mean, ema_std = {},{},{}
		best_val_loss_total = 1e9
		best_val_style_loss_total = 1e9
		epoch_start = 0
	

	for epoch in range(epoch_start,(epoch_start+n_epochs)):
		unet.train()
		epoch_loss = 0
		epoch_diff_loss = 0
		epoch_pix_loss = 0
		epoch_style_loss = 0
		epoch_mean_std_loss = 0
		epoch_soft_peak_loss = 0
		epoch_kl_loss = 0


		progress_bar = tqdm((enumerate(train_loader)),total=len(train_loader), ncols=150)
		progress_bar.set_description(f"Epoch {epoch}")
		for step, batch in progress_bar:
			gs +=1
			images = batch["image"].to(device)
			if image_min == -1.0:
				images = images * 2.0 - 1.0


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

			class_emb = batch['class_emb'].to(device)
			modality = int(class_emb.item())

			

			with autocast('cuda',enabled=True):
				# Initialize EMA mean/std if not present
				if modality not in ema_mean or modality not in ema_std:
					ref = (batch['image'].to(device) * 2.0 - 1.0)[brain_mask > 0] if image_min == -1.0 else batch['image'].to(device)[brain_mask > 0]

					ema_mean[modality] = ref.mean().detach()
					ema_std[modality] = ref.std().detach()
					if use_style_hist_01:
						ref01 = (ref + 1.0) / 2.0 if image_min == -1.0 else ref
						sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
						hist, centers_norm = soft_histogram(ref01, bins=hist_bins, value_range=(0.0,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
					else:
						sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
						hist, centers_norm = soft_histogram(ref, bins=hist_bins, value_range=(image_min,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
					
					ema_hist[modality] = hist.detach()

				# Generate random noise
				noise = torch.randn_like(images).to(device)

				# Create timesteps
				timesteps = torch.randint(
					0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
				).long()

				# Get model prediction
				if style_stats:
					noise_pred, noisy_image, snr_weight, latent_mean, latent_std = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
											condition=conditions, mode=condition_mode, class_label=class_emb, return_snr=use_snr_weight
										)

				else:
					inferer_outputs = inferer(
						inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
						condition=conditions, mode=condition_mode, class_label=class_emb, return_snr=use_snr_weight,
						ema_mean=ema_mean[modality], ema_std=ema_std[modality] # pass current global mean and std to UNet
					)

					# Always unpack the first two outputs
					noise_pred, noisy_image = inferer_outputs[:2]

					# Try to get snr_weight if present, else set to None or a default value
					snr_weight = inferer_outputs[2] if len(inferer_outputs) > 2 else None


				if use_snr_weight:
					loss_diff = F.mse_loss(noise_pred.float(), noise.float())
					loss_diff = torch.mean(loss_diff * snr_weight)
				else:
					loss_diff = F.mse_loss(noise_pred.float(), noise.float())
				if noisy_image.shape[1] != 1:
					noisy_image = noisy_image[:,:noisy_image.shape[1]//2,:,:,:] # since noisy_image returned are concat of noisy_image and condition


				if noise_loss == 'l2':
					x0_pred = torch.zeros_like(images).to(device)

					for n in range (len(noise_pred)):
						_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))
					
					pred_grad = util.torch_gradmap_average(x0_pred).float()
					org_grad = util.torch_gradmap_average(images).float()
					loss_grad = F.mse_loss(pred_grad, org_grad)

					if use_style_hist_01:
						x01 = (x0_pred + 1.0) / 2.0 if image_min == -1.0 else x0_pred
						sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
						hist, centers = soft_histogram(x01[brain_mask > 0], bins=hist_bins, value_range=(0.0,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
					else:
						sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
						hist, centers = soft_histogram(x0_pred[brain_mask > 0], bins=hist_bins, value_range=(image_min,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
					mean_pred = x0_pred[brain_mask > 0].mean()
					std_pred = x0_pred[brain_mask > 0].std()

					# Update EMA only if t < t_threshold
					if timesteps[0] < t_threshold:
						ema_hist[modality] = ema_hist[modality] * EMA_alpha + hist.detach() * (1-EMA_alpha)
						ema_mean[modality] = ema_mean[modality] * EMA_alpha + mean_pred.detach() * (1-EMA_alpha)
						ema_std[modality] = ema_std[modality] * EMA_alpha + std_pred.detach() * (1-EMA_alpha)

					# Soft peak loss (align current peak to EMA peak for this modality)
					peak_current, _ = soft_argmax(hist, centers.squeeze(0) if centers.dim() > 1 else centers)
					peak_ema, _ = soft_argmax(ema_hist[modality], centers.squeeze(0) if centers.dim() > 1 else centers)
					soft_peak_loss = F.mse_loss(peak_current, peak_ema)

					eps = 1e-8
					hist_norm = (hist + eps) / (hist.sum() + eps * hist.numel())
					ema_hist_norm = (ema_hist[modality] + eps) / (ema_hist[modality].sum() + eps * ema_hist[modality].numel())


					kl_loss = F.kl_div(hist_norm.log().unsqueeze(0), ema_hist_norm.unsqueeze(0), reduction='batchmean')
					style_loss = differentiable_wd(ema_hist[modality], hist)
					mean_std_loss = F.mse_loss(mean_pred, ema_mean[modality]) + F.mse_loss(std_pred, ema_std[modality])
					
					current_style_weight = get_style_weight(epoch, max_weight=style_weight, rampup_epochs=rampup_epochs) * max(0.0, 1.0 - timesteps[0] / t_threshold)
					current_mean_std_weight = mean_std_weight * max(0.0, 1.0 - timesteps[0] / t_threshold)
					current_peak_weight = peak_weight * max(0.0, 1.0 - timesteps[0] / t_threshold)

					loss = (loss_diff + content_weight * loss_grad 
					+ current_style_weight * style_loss 
					+ current_mean_std_weight * mean_std_loss
					+ current_peak_weight * soft_peak_loss
					)

				else:
					x0_pred = torch.zeros_like(images).to(device)

					for n in range (len(noise_pred)):
						_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))

					pred_grad = util.torch_gradmap_average(x0_pred).float()
					org_grad = util.torch_gradmap_average(images).float()
					loss_grad = F.mse_loss(pred_grad, org_grad)
					loss = loss_diff


			scaler.scale(loss).backward()
			del noise_pred, noisy_image, noise, images, x0_pred, conditions
			torch.cuda.empty_cache()

			# implement gradient accumulation
			if (step+1) % accumulation_steps == 0:
				scaler.step(optimizer_diff)
				scaler.update()
				optimizer_diff.zero_grad(set_to_none=True)

			epoch_loss += loss.item()
			epoch_diff_loss += loss_diff.item()
			epoch_pix_loss += loss_grad.item()
			epoch_style_loss += style_loss.item()
			epoch_mean_std_loss += mean_std_loss.item()
			epoch_soft_peak_loss += soft_peak_loss.item()
			epoch_kl_loss += kl_loss.item()


			del loss, loss_diff, loss_grad

			progress_bar.set_postfix({"loss": epoch_loss / (step + 1), "style_loss": epoch_style_loss / (step + 1)})
			
		writer.add_scalar('train_SGCD_step1/diffusion loss',epoch_diff_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/pixel loss',epoch_pix_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/total loss',epoch_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/style loss',epoch_style_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/mean_std loss',epoch_mean_std_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/soft peak loss',epoch_soft_peak_loss / (step + 1),epoch)
		writer.add_scalar('train_SGCD_step1/kl loss',epoch_kl_loss / (step + 1),epoch)

		if save_temp_ckp:
			temp_checkpoint_path = save_dir / 'TEMP_checkpoint.pth'
			torch.save({			
				'unet_state_dict': unet.state_dict(),
				'optimizer_diff': optimizer_diff.state_dict(),
				'scaler_state_dict': scaler.state_dict(),
				'lr_scheduler_state_dict': lr_scheduler.state_dict(),
				'epoch': epoch,
				'ema_hist': ema_hist,
				'ema_mean': ema_mean,
				'ema_std': ema_std,
				'best_val_loss_total': best_val_loss_total,
				'best_val_style_loss_total': best_val_style_loss_total,

			}, temp_checkpoint_path)



		if (epoch + 1) % val_interval == 0:
					
			unet.eval()
			val_loss_total = 0
			val_loss_diff_total = 0
			val_loss_pix_total = 0
			val_style_counter = 0
			val_style_loss_total = 0
			val_mean_std_loss_total = 0
			val_soft_peak_loss_total = 0
			val_kl_loss_total = 0

			with torch.no_grad():
				for (val_step, batch_val) in enumerate(val_loader, start=1):
					results = {}
					images = batch_val["image"].to(device)
					if image_min == -1.0:
						images = images * 2.0 - 1.0 # scale to -1, 1

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

					fn = batch_val['fn'][0]  # Get the filename of the source image for logging

					class_emb = batch_val['class_emb'].to(device)
					modality = int(class_emb.item())


				
					# Generate random noise
					noise = torch.randn_like(images).to(device)

					# Create timesteps
					timesteps = torch.randint(
						0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
					).long()

					# Get model prediction
					if style_stats:
						noise_pred, noisy_image, snr_weight, latent_mean, latent_std = inferer(
							inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
							condition=conditions, mode=condition_mode, class_label=class_emb, return_snr=use_snr_weight)
					else:
						inferer_outputs = inferer(
							inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps,
							condition=conditions, mode=condition_mode, class_label=class_emb, return_snr=use_snr_weight,
							ema_mean=ema_mean[modality], ema_std=ema_std[modality] # pass current global mean and std to UNet
						)

						# Always unpack the first two outputs
						noise_pred, noisy_image = inferer_outputs[:2]

						# Try to get snr_weight if present, else set to None or a default value
						snr_weight = inferer_outputs[2] if len(inferer_outputs) > 2 else None
	

					if use_snr_weight:
						val_loss_diff = F.mse_loss(noise_pred.float(), noise.float())
						val_loss_diff = torch.mean(val_loss_diff * snr_weight)
					else:
						val_loss_diff = F.mse_loss(noise_pred.float(), noise.float())

					if noisy_image.shape[1] != 1:
						noisy_image = noisy_image[:,:noisy_image.shape[1]//2,:,:,:] # since noisy_image returned are concat of noisy_image and condition


					if noise_loss == 'l2':
						x0_pred = torch.zeros_like(images).to(device)

						for n in range (len(noise_pred)):
							_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))

						pred_grad = util.torch_gradmap_average(x0_pred).float()
						org_grad = util.torch_gradmap_average(images).float()
						val_loss_grad = F.mse_loss(pred_grad, org_grad)

						if use_style_hist_01:
							x01 = (x0_pred + 1.0) / 2.0 if image_min == -1.0 else x0_pred
							sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
							hist, centers = soft_histogram(x01[brain_mask>0], bins=hist_bins, value_range=(0.0,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
						else:
							sigma_eff = hist_sigma * (t2_sigma if modality == 1 else 1.0)  # widen for T2
							hist, centers = soft_histogram(x0_pred[brain_mask>0], bins=hist_bins, value_range=(image_min,1.0), σ=sigma_eff, ignore=hist_ignore_bins)
						mean_pred = x0_pred[brain_mask > 0].mean()
						std_pred = x0_pred[brain_mask > 0].std()


						eps = 1e-8
						hist_norm = (hist + eps) / (hist.sum() + eps * hist.numel())
						ema_hist_norm = (ema_hist[modality] + eps) / (ema_hist[modality].sum() + eps * ema_hist[modality].numel())

						val_kl_loss = F.kl_div(hist_norm.log().unsqueeze(0), ema_hist_norm.unsqueeze(0), reduction='batchmean')

						val_style_loss = differentiable_wd(ema_hist[modality], hist)

						val_mean_std_loss = F.mse_loss(mean_pred, ema_mean[modality]) + F.mse_loss(std_pred, ema_std[modality])

						peak_current, _ = soft_argmax(hist, centers.squeeze(0) if centers.dim() > 1 else centers)
						peak_ema, _ = soft_argmax(ema_hist[modality], centers.squeeze(0) if centers.dim() > 1 else centers)
						soft_peak_loss = F.mse_loss(peak_current, peak_ema)

						current_style_weight = get_style_weight(epoch, max_weight=style_weight, rampup_epochs=rampup_epochs) * max(0.0, 1.0 - timesteps[0] / t_threshold)
						current_mean_std_weight = mean_std_weight * max(0.0, 1.0 - timesteps[0] / t_threshold)
						current_soft_peak_weight = peak_weight * max(0.0, 1.0 - timesteps[0] / t_threshold)
						val_loss = val_loss_diff + content_weight * val_loss_grad 
						+ current_style_weight * val_style_loss 
						+ current_mean_std_weight * val_mean_std_loss
						+ current_soft_peak_weight * soft_peak_loss

					
					else:
						x0_pred = torch.zeros_like(images).to(device)

						for n in range (len(noise_pred)):
							_, x0_pred[n] = scheduler.step(torch.unsqueeze(noise_pred[n,:,:,:,:], 0), timesteps[n], torch.unsqueeze(noisy_image[n,:,:,:,:], 0))

						val_loss_grad = F.l1_loss(images.float(),x0_pred.float())
						val_loss = val_loss_diff

					if val_step == 1:
						results['noisy_pred']=x0_pred.detach().cpu().float()

					
					del noise_pred, noisy_image, noise, x0_pred
					torch.cuda.empty_cache()


					# get the first sammple from the first validation batch for visualisation purposes
					if val_step == 1:
						tqdm.write('generating samples...')


						results['inputs']=images[:2].detach().cpu().float()
						results['conds']=conditions[:2].detach().cpu().float()

						
						if DDIM_inference:
							scheduler_ddim.set_timesteps(num_inference_steps=num_inference_fdp)
							img_noisy = inferer.reverse_sample(
								input_noise=images, diffusion_model=unet, scheduler=scheduler_ddim,
								conditioning=conditions,mode=condition_mode,verbose=False, class_label=class_emb,
								ema_mean=ema_mean[modality], ema_std=ema_std[modality]
							)
							results['noisy']=img_noisy.detach().cpu().float()

							scheduler_ddim.set_timesteps(num_inference_steps=num_inference_rdp)
							recon_images = inferer.sample(
								input_noise=img_noisy, diffusion_model=unet, scheduler=scheduler_ddim,
								conditioning=conditions,mode=condition_mode,verbose=False, class_label=class_emb,
								ema_mean=ema_mean[modality], ema_std=ema_std[modality]
								)
				
							results['recon']=recon_images.detach().cpu().float()
							del recon_images


						else:
							scheduler.set_timesteps(num_inference_steps=1000)
							recon_images = inferer.sample(
								input_noise=images[:2], diffusion_model=unet, scheduler=scheduler,
								conditioning=conditions, mode=condition_mode,class_label= class_emb,
								ema_mean=ema_mean[modality], ema_std=ema_std[modality]
								)
							results['recon']=recon_images.detach().cpu().float()
							del recon_images
							torch.cuda.empty_cache()


						results['error_recon']=torch.abs(results['inputs']-results['recon'])

						root = save_dir / 'images'/'val'
						if not root.exists():
							os.makedirs(root)
						for k in results:
							img_volume = results[k].detach().cpu() # torch.Size([batch, 1, 144, 184, 184])
							if image_min == -1.0:
								img_volume = (img_volume + 1.0) / 2.0 # -1,1 -> 0,1
							img_volume = torch.clamp(img_volume, 0, 1) # clamp to [0,1] range

							grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=False) # axial middle slices
							grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_a = (grid_a * 255).astype(np.uint8)
							filename = "{}_e-{:03}_t-{:04}_{}.png".format(k,epoch,int(timesteps[0]),'a')
							save_path = root / filename
							Image.fromarray(grid_a).save(save_path)


							grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=False) # coronal middle slice
							grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_c = (grid_c * 255).astype(np.uint8)
							filename = "{}_e-{:03}_t-{:04}_{}.png".format(k,epoch,int(timesteps[0]),'c')
							save_path = root / filename
							Image.fromarray(grid_c).save(save_path)

							grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=False) # saggital middle slice
							grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
							grid_s = (grid_s * 255).astype(np.uint8)
							filename = "{}_e-{:03}_t-{:04}_{}.png".format(k,epoch,int(timesteps[0]),'s')
							save_path = root / filename
							Image.fromarray(grid_s).save(save_path)

						del results, conditions

						np.save(root/f'ema_hist_{epoch}.npy', ema_hist)
						

					val_loss_total += val_loss.item()
					val_loss_diff_total += val_loss_diff.item()
					val_loss_pix_total += val_loss_grad.item()
					val_style_loss_total += val_style_loss.item()
					val_mean_std_loss_total += val_mean_std_loss.item()
					val_soft_peak_loss_total += soft_peak_loss.item()
					val_kl_loss_total += val_kl_loss.item()



					del val_loss, val_loss_diff, val_loss_grad
					

			val_loss_total /= val_step
			val_loss_diff_total /= val_step
			val_loss_pix_total /= val_step
			val_style_loss_total /= val_step
			val_mean_std_loss_total /= val_step
			val_soft_peak_loss_total /= val_step
			val_kl_loss_total /= val_step

			lr_scheduler.step(val_loss_total)
			writer.add_scalar('val_SGCD_step1/val total loss',val_loss_total,epoch)
			writer.add_scalar('val_SGCD_step1/val diffusion loss',val_loss_diff_total,epoch)
			writer.add_scalar('val_SGCD_step1/val pix loss',val_loss_pix_total,epoch)
			writer.add_scalar('val_SGCD_step1/val style loss',val_style_loss_total,epoch)
			writer.add_scalar('val_SGCD_step1/val mean_std loss',val_mean_std_loss_total,epoch)
			writer.add_scalar('val_SGCD_step1/val soft peak loss',val_soft_peak_loss_total,epoch)
			writer.add_scalar('val_SGCD_step1/val kl loss',val_kl_loss_total,epoch)

			now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

			# Save if val_loss_total is the best so far
			if val_loss_total < best_val_loss_total:
				best_val_loss_total = val_loss_total
				torch.save({
					'unet_state_dict': unet.state_dict(),
					'optimizer_diff': optimizer_diff.state_dict(),
					'scaler_state_dict': scaler.state_dict(),
					'lr_scheduler_state_dict': lr_scheduler.state_dict(),
					'epoch': epoch,
					'ema_hist': ema_hist,
					'ema_mean': ema_mean,
					'ema_std': ema_std,
					'best_val_loss_total': best_val_loss_total,
					'best_val_style_loss_total': best_val_style_loss_total,
				}, save_dir / f'BEST_ckp_valtotal_ep{epoch}.pth')
				tqdm.write(f'New best val_loss_total: {best_val_loss_total:.4f}. Saved BEST_ckp_valtotal_{now}.pth')
			# Save if val_style_loss_total is the best so far
			elif val_style_loss_total < best_val_style_loss_total:
				best_val_style_loss_total = val_style_loss_total
				torch.save({
					'unet_state_dict': unet.state_dict(),
					'optimizer_diff': optimizer_diff.state_dict(),
					'scaler_state_dict': scaler.state_dict(),
					'lr_scheduler_state_dict': lr_scheduler.state_dict(),
					'epoch': epoch,
					'ema_hist': ema_hist,
					'ema_mean': ema_mean,
					'ema_std': ema_std,
					'best_val_loss_total': best_val_loss_total,
					'best_val_style_loss_total': best_val_style_loss_total,
				}, save_dir / f'BEST_ckp_valstyle_ep{epoch}.pth')
				tqdm.write(f'New best val_style_loss_total: {best_val_style_loss_total:.4f}. Saved BEST_ckp_valstyle_{now}.pth')




if __name__ == "__main__":
	if torch.cuda.is_available():
		device = torch.cuda.current_device()
		print('Using CUDA: ',torch.cuda.get_device_name(device))
	else:
		print("CUDA is not available.")
	torch.cuda.empty_cache()

	### training parameters
	n_epochs = 300
	bs = 1
	accumulation_steps = 4 # effective batch size = batch size * accumulation_steps
	val_interval = 10 
	lr_patience = 2

	noise_loss = 'l2'
	use_snr_weight = False
	norm = 'AdaIN'
	content_weight = 1
	style_weight = 10 
	mean_std_weight = 1 
	peak_weight = 1 
	EMA_alpha = 0.95 
	rampup_epochs = 20 
	t_threshold = 100 # threshold for ema update
	condition_on = 'grad' # 'grad' or None


	### histogram parameters
	use_style_hist_01 = True
	hist_bins = 100
	image_min = -1.0 # set the input image to [-1,1] range
	bin_width = 0.01
	t2_sigma = 1.5 
	hist_sigma = bin_width
	hist_ignore_bins = int(round(0.02 / bin_width))
	

	### DDIM Sampling parameters
	DDIM_inference = True
	DDIM_train = False
	clip_sample = True
	num_train_ddim = 50
	num_inference_fdp = 35
	num_inference_rdp = 25

	### debug parameters
	save_script_snapshot = True
	style_stats = False



	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
	run_name = 'DEFINE_YOUR_RUN_NAME' 
	

	print(run_name)





	Resume = False
	if Resume:
		resume_from = Path('PATH_TO_YOUR_CHECKPOINT.pth')  # specify your checkpoint path here
		print(f'Resume from: {resume_from}')
		save_dir = resume_from.parent
		writer=SummaryWriter(f'TBLOG/{save_dir.name}')
	else:
		epoch_start = 0
		writer=SummaryWriter(f'TBLOG/{now}_{run_name}')
		save_dir = Path(f'log/Stage1/{now}_{run_name}')
  
	if not save_dir.exists():
		os.makedirs(save_dir)
	elif not Resume:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'
  



	# ############################## Define Dataset and DataLoader  ########################################################### 
	data_pt = Path('PATH_TO_YOUR_DATA_DIRECTORY')  # specify your data path here

	# combine source and target
	lb_train_combined = pd.concat([
									pd.read_csv('PATH_TO_TRAIN_T1.tsv',sep='\t'),
									pd.read_csv('PATH_TO_TRAIN_T2.tsv',sep='\t')
									])
	
	lb_val_combined = pd.concat([
								  pd.read_csv('PATH_TO_VAL_T1.tsv',sep='\t'),
								  pd.read_csv('PATH_TO_VAL_T2.tsv',sep='\t')
								  ])
	

	train_dataset = MRI.DWITHP_t1t2(data_pt, lb_train_combined,image_min=image_min)
	val_dataset = MRI.DWITHP_t1t2(data_pt, lb_val_combined,image_min=image_min)
	train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True,drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True,drop_last=True)
	brain_mask = torch.from_numpy(np.load('PATH_TO_YOUR_BRAIN_MASK_FILE.npy')).unsqueeze(0).float().to(device) # (1,1,H,W,D)



	if len(train_loader) > 70:
		save_temp_ckp = True
	else:
		save_temp_ckp = False





	# define model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using {device}")

	parameters = f"""
	n_epochs = {n_epochs},
	pix_loss = {noise_loss},
	content_weight = {content_weight},
	use_snr_weight = {use_snr_weight},
	DDIM_inference = {DDIM_inference},
	DDIM_train = {DDIM_train},
	num_train_ddim = {num_train_ddim},
	num_inference_fdp = {num_inference_fdp},
	num_inference_rdp = {num_inference_rdp},
	run_name = {run_name},
	save_temp_ckp = {save_temp_ckp},
	norm = {norm},
	val_interval = {val_interval},
	accumulation_steps = {accumulation_steps},
	style_stats = {style_stats},
	style_weight = {style_weight},
	mean_std_weight = {mean_std_weight},
	rampup_epochs = {rampup_epochs},
	condition_on = {condition_on},
	clip_sample during training = {clip_sample},
	t_threshold = {t_threshold},
	peak_weight = {peak_weight},
	EMA_alpha = {EMA_alpha},
	hist_bins = {hist_bins},
	use_style_hist_01 = {use_style_hist_01},
	hist_sigma = {hist_sigma},
	lr_patience = {lr_patience}
	"""
 
	print(parameters)
	with open(save_dir / 'parameters.txt', 'w') as f:
		f.write(run_name+'\n')
		f.write(parameters)

	# Save a snapshot of the running script
	# if not (Resume or val_only):
	if save_script_snapshot:
		try:
			script_path = Path(__file__).resolve()
			destination_path = save_dir / f"{script_path.stem}_{now}.py"
			shutil.copy2(script_path, destination_path)
			print(f"Saved script snapshot to {destination_path}")
		except NameError:
			print("Could not save script snapshot: `__file__` is not defined (e.g., in an interactive session).")
		except Exception as e:
			print(f"Error saving script snapshot: {e}")

	
	train()