#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: infer_Stage1.py
Description: inference script for the MMH Stage I: Sequence-Specific Global Harmonization

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
from pathlib import Path
import datetime
import data.MRIdata as MRI
import numpy as np
import torch
import torchvision
from PIL import Image
import pandas as pd
from monai.data import DataLoader
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
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



def inference():
	with torch.inference_mode():
		for (val_step, batch) in progress_bar: # source only
			results = {}
			fn = batch['fn']
			fn_tar = batch['tar_fn']

			if resume and all(item in resume_fn_list for item in fn):
				continue

			images = batch["image"].to(device)
			if image_min == -1:
				images = images * 2.0 - 1.0 # scale to [-1,1]

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

			

	
			if not (conditions is None):
				if conditions.shape[0] < images.shape[0]:
					repeats = images.shape[0] // conditions.shape[0]
					conditions = conditions.repeat(repeats, 1, 1, 1, 1)
				else:
					conditions = conditions[:len(images)]

			results['input']=images.detach().cpu().float()

			if sch == 'DDPM':
				scheduler.set_timesteps(num_inference_steps=ddpm_step) # DDPM
			else:
				scheduler.set_timesteps(num_inference_steps=num_inference_fdp) # DDIM

			# reverse DDIM sampling to add noise
			img_noisy = inferer.reverse_sample(
				input_noise=images, diffusion_model=unet, scheduler=scheduler,
				conditioning=conditions,mode=condition_mode,verbose=True, class_label=class_emb,
				ema_mean=ema_mean_batch, ema_std=ema_std_batch
			)
			results['noisy']=img_noisy.detach().cpu().float()

			scheduler.set_timesteps(num_inference_steps=num_inference_rdp)
			recon_images = inferer.sample(
				input_noise=img_noisy, diffusion_model=unet, scheduler=scheduler,
				conditioning=conditions,mode=condition_mode,verbose=True, class_label=class_emb,
				ema_mean=ema_mean_batch, ema_std=ema_std_batch
				)
   
			results['recon']=recon_images.detach().cpu().float()
			del recon_images
			torch.cuda.empty_cache()
			
			
			if save_volume:
				for b_idx in range(len(fn)):
					for k in results:
						img_volume = results[k][b_idx].detach().cpu() # [1,W,H,Z], eg: torch.Size([1, 184, 184, 184])
						if image_min == -1.0:
							img_volume = (img_volume + 1.0) / 2.0 # -1,1 -> 0,1
						img_volume = torch.clamp(img_volume, min=0.0, max=1.0)
						if k == 'condition':
							save_fn = f'{fn_tar[b_idx]}_{k}.npy'
						else:
							save_fn = f'{fn[b_idx]}_{k}.npy'
						full_save_pt = save_dir/save_fn
						np.save(full_save_pt,img_volume.squeeze().float())

			if save_sample:
				root = save_dir / 'samples'
				if not root.exists():
					os.makedirs(root)
				for k in results:
					img_volume = results[k].detach().cpu() 
					img_volume = torch.clamp(img_volume, min=0.0, max=1.0)


					grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1) # axial middle slices
					grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_a = (grid_a * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'a')
					save_path = root / filename
					Image.fromarray(grid_a).save(save_path)


					grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4) # coronal middle slice
					grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_c = (grid_c * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'c')
					save_path = root / filename
					Image.fromarray(grid_c).save(save_path)

					grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4) # saggital middle slice
					grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
					grid_s = (grid_s * 255).astype(np.uint8)
					filename = "{}_{}_{}.png".format(fn[0],k,'s')
					save_path = root / filename
					Image.fromarray(grid_s).save(save_path)


if __name__ == '__main__':
	bs = 4
	if torch.cuda.is_available():
		device = torch.cuda.current_device()
		print('Using CUDA: ',torch.cuda.get_device_name(device))
	else:
		print("CUDA is not available.")
	torch.cuda.empty_cache()

	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")


	


	save_volume = True
	save_sample = False
	save_intermediate = False

	norm = 'AdaIN'
	condition_on = 'grad'  # grad or None

	image_min = -1

 
	sch = 'DDIM'

	num_train_ddim = 50
	num_inference_fdp = 35
	num_inference_rdp = 25
 
	ddpm_step = 100

	run_name = 'DEFINE_YOUR_RUN_NAME' 
	

	print(run_name)

	save_dir = Path(f'PATH_TO_SAVE_INFERENCE_RESULTS')

	resume = False
	if resume:
		resume_dir = Path(f'PATH_TO_SAVE_INFERENCE_RESULTS')
		save_dir = resume_dir
		print(f'Resuem inference, find {len(os.listdir(save_dir))} files')
		resume_fn_list = [f.replace('_recon.npy','') for f in os.listdir(save_dir)] 

	elif not save_dir.exists():
		os.makedirs(save_dir)
	elif not resume:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'

	############################## Define Dataset and DataLoader  ########################################################### 
	data_pt = Path('PATH_TO_YOUR_DATA_DIRECTORY')  # specify your data path here

	# combine source and target
	
	lb_test_combined = pd.concat([
								  pd.read_csv('PATH_TO_TEST_T1.tsv',sep='\t'),
								  pd.read_csv('PATH_TO_TEST_T2.tsv',sep='\t')
								  ])

	test_dataset = MRI.DWITHP_t1t2(data_pt, lb_test_combined,image_min=image_min)
	test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True,drop_last=True)
	brain_mask = torch.from_numpy(np.load('PATH_TO_YOUR_BRAIN_MASK_FILE.npy')).unsqueeze(0).float().to(device) # (1,1,H,W,D)

		

	#### load pth
	DDPM_pt = torch.load(f'PATH_TO_BEST_STAGE1_CKP.pth') # load the best stage1 checkpoint


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

	unet.load_state_dict(DDPM_pt['unet_state_dict'])
	print('DDPM weighted loaded!')
	EMA_mean = DDPM_pt['ema_mean']
	EMA_std = DDPM_pt['ema_std']
	print('EMA mean and std loaded!:',EMA_mean, EMA_std)

	progress_bar = tqdm((enumerate(test_loader)),total=len(test_loader), ncols=150)
	progress_bar.set_description(f"Inference Source")

	for param in unet.parameters():
		param.requires_grad = False
	unet.eval()

	if sch == 'DDPM':
		scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
	else:
		scheduler = DDIMScheduler(
			num_train_timesteps=num_train_ddim, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195, clip_sample=False
		)
	inferer = DiffusionInferer(scheduler)
	
	inference()