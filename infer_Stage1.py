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

License: Apache-2.0 License (see LICENSE file for details)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from pathlib import Path
import datetime
import argparse
import MRIdata as MRI
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
	parser = argparse.ArgumentParser(description="MMH Stage I Inference")
		
	parser.add_argument('--bs', type=int, default=4, help='Batch size')
	parser.add_argument('--disable_save_volume', action='store_true', help='Disable saving volumes')
	parser.add_argument('--save_sample', action='store_true', help='Enable saving samples')
	parser.add_argument('--save_intermediate', action='store_true', help='Enable saving intermediate results')
	parser.add_argument('--norm', type=str, default='AdaIN', help='Normalization string')
	parser.add_argument('--condition_on', type=str, default='grad', help='Condition on: grad or none')
	parser.add_argument('--image_min', type=float, default=-1.0, help='Image min value')
	parser.add_argument('--sch', type=str, default='DDIM', help='Scheduler type: DDIM or DDPM')
	parser.add_argument('--num_train_ddim', type=int, default=50, help='Num train DDIM steps')
	parser.add_argument('--num_inference_fdp', type=int, default=35, help='Num inference FDP steps')
	parser.add_argument('--num_inference_rdp', type=int, default=25, help='Num inference RDP steps')
	parser.add_argument('--ddpm_step', type=int, default=100, help='DDPM steps')
	parser.add_argument('--run_name', type=str, default='DEFINE_YOUR_RUN_NAME', help='Run name')
	parser.add_argument('--save_dir', type=str, default='PATH_TO_SAVE_INFERENCE_RESULTS', help='Path to save inference results')
	parser.add_argument('--resume', action='store_true', help='Resume inference')
	parser.add_argument('--resume_dir', type=str, default='PATH_TO_SAVE_INFERENCE_RESULTS', help='Resume directory')
	parser.add_argument('--data_pt', type=str, default='PATH_TO_YOUR_DATA_DIRECTORY', help='Path to data directory')
	parser.add_argument('--test_tsvs', nargs='+', default=['PATH_TO_TEST_T1.tsv', 'PATH_TO_TEST_T2.tsv'], help='One or multiple paths to test TSV files')
	parser.add_argument('--brain_mask', type=str, default='PATH_TO_YOUR_BRAIN_MASK_FILE.npy', help='Path to brain mask')
	parser.add_argument('--stage1_model', type=str, default='PATH_TO_BEST_STAGE1_CKP.pth', help='Path to Stage 1 checkpoint')

	args = parser.parse_args()

	bs = args.bs
	save_volume = not args.disable_save_volume
	save_sample = args.save_sample
	save_intermediate = args.save_intermediate
	norm = args.norm
	condition_on = args.condition_on if args.condition_on.lower() != 'none' else None
	image_min = args.image_min
	sch = args.sch
	num_train_ddim = args.num_train_ddim
	num_inference_fdp = args.num_inference_fdp
	num_inference_rdp = args.num_inference_rdp
	ddpm_step = args.ddpm_step
	run_name = args.run_name
	save_dir = Path(args.save_dir)
	resume = args.resume
	resume_dir = Path(args.resume_dir)
	data_pt = Path(args.data_pt)

	if torch.cuda.is_available():
		device = torch.cuda.current_device()
		print('Using CUDA: ',torch.cuda.get_device_name(device))
	else:
		print("CUDA is not available.")
	torch.cuda.empty_cache()

	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
	
	print(run_name)

	if resume:
		save_dir = resume_dir
		print(f'Resume inference, find {len(os.listdir(save_dir))} files')
		resume_fn_list = [f.replace('_recon.npy','') for f in os.listdir(save_dir)] 

	elif not save_dir.exists():
		os.makedirs(save_dir)
	elif not resume:
		assert len(os.listdir(save_dir))==0,'Log dir exist!'

	############################## Define Dataset and DataLoader  ########################################################### 
	
	# combine tsv sources
	lb_test_combined = pd.concat([pd.read_csv(tsv_path, sep='\t') for tsv_path in args.test_tsvs], ignore_index=True)

	test_dataset = MRI.DWITHP_t1t2(data_pt, lb_test_combined,image_min=image_min)
	test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True,drop_last=True)
	brain_mask = torch.from_numpy(np.load(args.brain_mask)).unsqueeze(0).float().to(device) # (1,1,H,W,D)
		
	#### load pth
	DDPM_pt = torch.load(args.stage1_model) # load the best stage1 checkpoint


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