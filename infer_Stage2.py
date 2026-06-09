#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: infer_Stage2.py
Description: inference script for the MMH Stage 2: Target-Specific Fine Harmonization

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
from pathlib import Path
import datetime
import argparse
import numpy as np
import torch
from torch.amp import autocast
import pandas as pd
from tqdm import tqdm
from monai import transforms
from monai.data import DataLoader, MetaTensor
from monai.utils import set_determinism
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler
from monai.inferers import DiffusionInferer
import MRIdata as MRI
import util
import torchvision
from PIL import Image

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
set_determinism(seed=SEED)

class StyleRmNpyDataset(torch.utils.data.Dataset):
	def __init__(self, root, df):
		self.root = Path(root)
		self.fns = [Path(p).name.replace(".nii.gz", "") for p in df["ImageFilePath"]]

	def __len__(self):
		return len(self.fns)

	def __getitem__(self, idx):
		fn = self.fns[idx]
		img = np.load(self.root / f"{fn}.npy").astype(np.float32)
		img = torch.from_numpy(img)
		if img.ndim == 3:
			img = img.unsqueeze(0)
		return {"image": img, "fn": fn, "class_emb": torch.tensor(0)}

def visualize_batch_4D_tensor(img_volume, fn, save_pt, prefix=''):
	#### set normalize to Falase to reflect the true pixel values of MRI slices
	os.makedirs(save_pt, exist_ok=True)
	print(f"Visualizing {fn} with shape {img_volume.shape}")
	img_volume = img_volume.reshape(-1, 1, *img_volume.shape[-3:])
	print(f'original min: {img_volume.min().item()}, max: {img_volume.max().item()}, mean: {img_volume.mean().item()}')
		

	if image_min == -1.0 and img_volume.min() < 0:
		img_volume = torch.clamp(img_volume,image_min,1)
		img_volume = (img_volume + 1.0) / 2.0  # [-1,1] -> [0,1]
	else:
		img_volume = torch.clamp(img_volume,0,1)
	# img_volume = (img_volume-img_volume.min())/(img_volume.max()-img_volume.min()) # -> [0,1]

	print(f'after clip min: {img_volume.min().item()}, max: {img_volume.max().item()}, mean: {img_volume.mean().item()}')
	grid_a = torchvision.utils.make_grid(img_volume[:,:,:,:,img_volume.shape[4]//2], nrow=1,normalize=False, value_range=(0,1)) # axial middle slices
	grid_a = grid_a.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_a = (grid_a * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'a')
	save_path = save_pt / filename
	Image.fromarray(grid_a).save(save_path)


	grid_c = torchvision.utils.make_grid(img_volume[:,:,:,img_volume.shape[3]//2,:], nrow=4,normalize=False, value_range=(0,1)) # coronal middle slice
	grid_c = grid_c.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_c = (grid_c * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'c')
	save_path = save_pt / filename
	Image.fromarray(grid_c).save(save_path)

	grid_s = torchvision.utils.make_grid(img_volume[:,:,img_volume.shape[2]//2,:,:], nrow=4,normalize=False, value_range=(0,1)) # saggital middle slice
	grid_s = grid_s.transpose(0, 1).transpose(1, 2).squeeze(-1).rot90().numpy()
	grid_s = (grid_s * 255).astype(np.uint8)
	filename = "{}{}_{}.png".format(prefix,fn,'s')
	save_path = save_pt / filename
	Image.fromarray(grid_s).save(save_path)

def _as_tensor(x):
	return x.as_tensor() if isinstance(x, MetaTensor) else x


def match_mean_std(vol, ref, eps=1e-9, image_min=None, clamp=True):
	vol_mean = vol.mean()
	vol_std = vol.std()
	ref_mean = ref.mean()
	ref_std = ref.std()
	vol = (vol - vol_mean) / (vol_std + eps)
	out = vol * (ref_std + eps) + ref_mean
	if clamp:
		rmin = -1.0 if image_min == -1.0 else 0.0
		out = torch.clamp(out, rmin, 1.0)
	return out


def scale_images(data_dir, target_image=None, image_min=0.0):
	for img_path in Path(data_dir).glob("*.npy"):
		img = np.load(img_path)


		ref = np.load(target_image)
		img_scaled = match_mean_std(torch.from_numpy(img), torch.from_numpy(ref), image_min=image_min)
		np.save(img_path, img_scaled.numpy())


def _ema_stats(class_emb, ema_mean, ema_std, device):
	if class_emb is None or ema_mean is None or ema_std is None:
		return None, None
	try:
		cls_list = [int(x) for x in class_emb.view(-1).cpu().tolist()]
	except Exception:
		cls_list = [int(class_emb.item())]
	mean_batch = torch.tensor([ema_mean[c] for c in cls_list], device=device, dtype=torch.float32)
	std_batch = torch.tensor([ema_std[c] for c in cls_list], device=device, dtype=torch.float32)
	return mean_batch, std_batch


def _build_conditions(batch, condition_on, grad_cond_type, device):
	if condition_on == "self":
		conditions = _as_tensor(batch["image"]).to(device)
	elif condition_on == "target":
		conditions = _as_tensor(batch["target"]).to(device)
	elif condition_on == "grad":
		conditions = _as_tensor(batch["image"]).to(device)
		if grad_cond_type == "avg_norm":
			conditions = util.torch_gradmap_average(conditions)
			conditions = util.norm_gradmap_percnetile(conditions)
		elif grad_cond_type == "sobel":
			conditions = util.torch_sobelmap_3d(conditions)
		elif grad_cond_type == "canny":
			conditions = util.real_canny_map_3d(conditions)
		elif grad_cond_type == "freq":
			conditions = util.freq_cond(conditions)
		conditions = torch.tanh(conditions.clamp(-10.0, 10.0)) * 0.5
	else:
		conditions = None
	return conditions


def fdp(inferer, scheduler_ddim, model, input_img, class_emb, conditions, mode, ema_mean, ema_std, steps):
	scheduler_ddim.set_timesteps(num_inference_steps=steps)
	return inferer.reverse_sample(
		input_noise=input_img,
		diffusion_model=model,
		scheduler=scheduler_ddim,
		conditioning=conditions,
		mode=mode,
		verbose=False,
		class_label=class_emb,
		ema_mean=ema_mean,
		ema_std=ema_std,
	)


def rdp(inferer, scheduler_ddim, model, latent_noisy, class_emb, conditions, mode, ema_mean, ema_std, steps):
	scheduler_ddim.set_timesteps(num_inference_steps=steps)
	return inferer.sample(
		input_noise=latent_noisy,
		diffusion_model=model,
		scheduler=scheduler_ddim,
		conditioning=conditions,
		mode=mode,
		verbose=False,
		class_label=class_emb,
		ema_mean=ema_mean,
		ema_std=ema_std,
	)


def style_removal(
	loader,
	unet,
	scheduler_ddim,
	inferer,
	save_dir,
	condition_on,
	grad_cond_type,
	ema_mean,
	ema_std,
	image_min,
	num_inference_fdp,
	num_inference_rdp,
	use_amp,
):
	out_dir = save_dir / "1_style_removed_images"
	out_dir.mkdir(parents=True, exist_ok=True)

	with torch.inference_mode():
		for batch in tqdm(loader, desc="Style removal", dynamic_ncols=True):
			image = _as_tensor(batch["image"]).to(device)
			fn_list = batch["fn"]
			class_emb = batch.get("class_emb", None)
			if class_emb is not None:
				class_emb = class_emb.to(device)
			ema_mean_batch, ema_std_batch = _ema_stats(class_emb, ema_mean, ema_std, device)

			conditions = _build_conditions(batch, condition_on, grad_cond_type, device)
			mode = "concat" if conditions is not None else "crossattn"

			with autocast("cuda", enabled=use_amp):
				img_noisy = fdp(
					inferer,
					scheduler_ddim,
					unet,
					image,
					class_emb,
					conditions,
					mode,
					ema_mean_batch,
					ema_std_batch,
					num_inference_fdp,
				)
				recon = rdp(
					inferer,
					scheduler_ddim,
					unet,
					img_noisy,
					class_emb,
					conditions,
					mode,
					ema_mean_batch,
					ema_std_batch,
					num_inference_rdp,
				)

			for idx, recon_img in enumerate(recon):
				if image_min == -1.0:
					recon_img = torch.clamp(recon_img, -1.0, 1.0)
				else:
					recon_img = torch.clamp(recon_img, 0.0, 1.0)
				np.save(out_dir / f"{fn_list[idx]}.npy", recon_img.detach().cpu().float().numpy())
			
			visualize_batch_4D_tensor(recon.detach().cpu().float(), 
							'_'.join([f for f in fn_list]).replace('THP000',''), out_dir/'sample_style_removed')
			visualize_batch_4D_tensor(batch['image'].detach().cpu().float(), 
							'_'.join([f for f in fn_list]).replace('THP000',''), out_dir/'sample_org')


def pre_compute_latent(
	loader,
	unet,
	scheduler_ddim,
	inferer,
	save_dir,
	condition_on,
	grad_cond_type,
	ema_mean,
	ema_std,
	num_inference_fdp,
	use_amp,
):
	out_dir = save_dir / "2_fdp_latents"
	out_dir.mkdir(parents=True, exist_ok=True)

	with torch.inference_mode():
		for batch in tqdm(loader, desc="Pre-compute latents", dynamic_ncols=True):
			images = _as_tensor(batch["image"]).to(device)
			fn_list = batch["fn"]
			class_emb = batch.get("class_emb", None)
			if class_emb is not None:
				class_emb = class_emb.to(device)
			ema_mean_batch, ema_std_batch = _ema_stats(class_emb, ema_mean, ema_std, device)

			conditions = _build_conditions(batch, condition_on, grad_cond_type, device)
			mode = "concat" if conditions is not None else "crossattn"

			with autocast("cuda", enabled=use_amp):
				latent_noisy = fdp(
					inferer,
					scheduler_ddim,
					unet,
					images,
					class_emb,
					conditions,
					mode,
					ema_mean_batch,
					ema_std_batch,
					num_inference_fdp,
				)

			for idx, recon_latent in enumerate(latent_noisy):
				torch.save(
					{
						"latent": recon_latent.detach().cpu().float(),
						"condition": conditions[idx].detach().cpu().float() if conditions is not None else None,
						"class_emb": class_emb[idx].detach().cpu() if class_emb is not None else None,
					},
					out_dir / f"{fn_list[idx]}.pt",
				)


class LatentDataset(torch.utils.data.Dataset):
	def __init__(self, style_rm_dir, latent_dir, labels, transform=None):
		self.style_rm_dir = Path(style_rm_dir)
		self.latent_dir = Path(latent_dir)
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		fn = str(self.labels.iloc[idx, 0])
		style_rm_files = list(self.style_rm_dir.glob(f"*{fn}.*"))
		if not style_rm_files:
			raise FileNotFoundError(f"No style-removed image for {fn} in {self.style_rm_dir}")
		style_rm_path = style_rm_files[0]
		if style_rm_path.suffix == ".npy":
			style_rm_img = torch.from_numpy(np.load(style_rm_path)).float()
		else:
			style_rm_img = torch.load(style_rm_path, weights_only=False).float()

		if self.transform is not None:
			style_rm_img = self.transform(style_rm_img)
		style_rm_img = _as_tensor(style_rm_img)

		latent_files = list(self.latent_dir.glob(f"*{fn}.pt"))
		if not latent_files:
			raise FileNotFoundError(f"No latent file for {fn} in {self.latent_dir}")
		latent_pack = torch.load(latent_files[0], weights_only=False)
		latent = latent_pack["latent"].float()
		condition = latent_pack.get("condition", None)
		class_emb = latent_pack.get("class_emb", None)
		if condition is not None:
			condition = condition.float()
		if latent.ndim != 4:
			latent = latent.unsqueeze(0)

		return {
			"latent": latent,
			"condition": condition,
			"style_rm_img": style_rm_img,
			"fn": fn,
			"class_emb": class_emb,
		}


def _resolve_target_path(data_pt_val, fn, target_site, class_label):
	subject = fn.split("_")[0]
	suffix = ""
	if class_label is not None:
		try:
			suffix = "_T2" if int(class_label.item()) == 1 else ""
		except Exception:
			suffix = ""
	direct = data_pt_val / f"{subject}_{target_site}{suffix}.npy"
	if direct.exists():
		return direct
	candidates = list(data_pt_val.glob(f"{subject}*{target_site}*{suffix}*.npy"))
	if not candidates:
		raise FileNotFoundError(f"No target image found for {fn} under {data_pt_val}")
	return candidates[0]


def CLIP_val(
	model,
	val_loader,
	target_image_dir,
	target_site,
	transform,
	brain_mask,
	scheduler_ddim,
	inferer,
	ema_mean,
	ema_std,
	image_min,
	num_inference_rdp,
	save_dir,
	save_outputs=True,
	use_amp=True,
):
	model.eval()
	val_wd_total = 0.0
	count = 0
	out_dir = save_dir / "5_infer"
	out_dir.mkdir(parents=True, exist_ok=True)

	with torch.inference_mode():
		for batch in tqdm(val_loader, desc="Inference", dynamic_ncols=True):
			conditioning = batch["condition"]
			if conditioning is not None:
				conditioning = conditioning.to(device)
			class_label = batch.get("class_emb", None)
			if class_label is not None:
				class_label = class_label.to(device)
			ema_mean_batch, ema_std_batch = _ema_stats(class_label, ema_mean, ema_std, device)

			latent = batch["latent"].to(device)
			fn_list = batch["fn"]
			mode = "concat" if conditioning is not None else "crossattn"

			# target_path = _resolve_target_path(data_pt_val, fn_list[0], target_site, class_label)
			target_image = transform(torch.from_numpy(np.load(target_image_dir))).unsqueeze(0).float().to(device)
			target_image = _as_tensor(target_image)

			with autocast("cuda", enabled=use_amp):
				recon = rdp(
					inferer,
					scheduler_ddim,
					model,
					latent,
					class_label,
					conditioning,
					mode,
					ema_mean_batch,
					ema_std_batch,
					num_inference_rdp,
				)

			recon = match_mean_std(recon, target_image, image_min=image_min)
			hist_pred, _ = util.soft_histogram(
				recon[brain_mask > 0], 100, (image_min, 1.0), 0.01, 2
			)
			hist_target, _ = util.soft_histogram(
				target_image[brain_mask > 0], 100, (image_min, 1.0), 0.01, 2
			)
			val_wd_total += util.differentiable_wd(hist_pred, hist_target).item()
			count += 1

			if save_outputs:
				for idx, recon_img in enumerate(recon):
					if image_min == -1.0:
						recon_img = (recon_img + 1.0) / 2.0
						recon_img = recon_img.clamp(0.0, 1.0)
					else:
						recon_img = recon_img.clamp(0.0, 1.0)
					recon_img = transform_output(recon_img)
					np.save(out_dir / f"{fn_list[idx]}.npy", recon_img.cpu().numpy())

	return val_wd_total / max(1, count)


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	parser = argparse.ArgumentParser(description="MMH Stage II Inference")
	now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

	parser.add_argument('--save_dir', type=str, default='PATH_TO_SAVE_INFERENCE_RESULTS', help='Path to save inference results')
	parser.add_argument('--data_pt', type=str, default='PATH_TO_YOUR_DATA_DIRECTORY', help='Path to data directory')
	parser.add_argument('--test_tsvs', nargs='+', default=['PATH_TO_TEST_T1.tsv', 'PATH_TO_TEST_T2.tsv'], help='One or multiple paths to test TSV files')
	parser.add_argument('--stage1_model', type=str, default='PATH_TO_BEST_STAGE1_CKP.pth', help='Path to Stage 1 checkpoint')
	parser.add_argument('--stage2_model', type=str, default='PATH_TO_BEST_STAGE2_CKP.pth', help='Path to Stage 2 checkpoint')
	parser.add_argument('--target_image', type=str, default='PATH_TO_TARGET_IMAGE.npy', help='Path to a sample target image (from the target domain where the stage 2 model was trained) for optional visualization and WD calculation')
	parser.add_argument('--target_site', type=str, default='SITE_LABEL', help='Target site identifier used in filenames')
	parser.add_argument('--brain_mask', type=str, default='PATH_TO_YOUR_BRAIN_MASK_FILE.npy', help='Path to brain mask')



	parser.add_argument('--num_train_ddim', type=int, default=50, help='Num train DDIM steps')
	parser.add_argument('--num_inference_fdp', type=int, default=35, help='Num inference FDP steps')
	parser.add_argument('--num_inference_rdp', type=int, default=25, help='Num inference RDP steps')
	parser.add_argument('--image_min', type=float, default=-1.0, help='Image min value')
	parser.add_argument('--condition_on', type=str, default='grad', help='Condition on: grad or none')



	args = parser.parse_args()

	run_name = "Stage2_inference"
	save_dir = Path(args.save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)

	num_train_ddim = args.num_train_ddim
	num_inference_fdp = args.num_inference_fdp
	num_inference_rdp = args.num_inference_rdp
	image_min = args.image_min
	use_mask = True
	use_amp = True
	apply_scale_images = True

	condition_on = args.condition_on
	grad_cond_type = "avg_norm"

	data_pt = Path(args.data_pt)
	target_image = args.target_image
	target_site = args.target_site
	brain_mask_path = args.brain_mask

	lb_inference = pd.concat([pd.read_csv(tsv_path, sep='\t') for tsv_path in args.test_tsvs], ignore_index=True)


	transform = transforms.Compose(
		[
			transforms.CenterSpatialCrop(roi_size=(144, 184, 184)),
			transforms.ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=image_min, b_max=1.0, clip=True),
		]
	)

	transform_output = transforms.Compose(
		[
			transforms.SpatialPad(spatial_size=(184, 184, 184)), # transform to 184^3 for visualization 
		]
	)

	if use_mask:
		brain_mask = torch.from_numpy(np.load(brain_mask_path)).unsqueeze(0).float().to(device)
	else:
		brain_mask = torch.ones((1, 1, 144, 184, 184), device=device)

	stage1_ckp_pt = Path(args.stage1_model)
	stage1_ckp = torch.load(stage1_ckp_pt, map_location="cpu")
	
	stage2_ckp_pt = Path(args.stage2_model)
	stage2_ckp = torch.load(stage2_ckp_pt, map_location="cpu") 

	unet = DiffusionModelUNet(
		spatial_dims=3,
		in_channels=2,
		out_channels=1,
		num_res_blocks=2,
		channels=(32, 64, 256, 256),
		attention_levels=(False, False, True, True),
		num_head_channels=(0, 0, 32, 32),
		norm_num_groups=16,
		use_flash_attention=True,
		num_class_embeds=2,
		norm="AdaIN",
	).to(device)
	unet.load_state_dict(stage1_ckp["unet_state_dict"])
	unet.eval()
	print("Stage 1 model loaded.")

	ema_mean = stage1_ckp.get("ema_mean", None)
	ema_std = stage1_ckp.get("ema_std", None)

	scheduler_ddim = DDIMScheduler(
		num_train_timesteps=num_train_ddim,
		schedule="scaled_linear_beta",
		beta_start=0.0015,
		beta_end=0.0195,
		clip_sample=False,
	)
	inferer = DiffusionInferer(scheduler_ddim)

	styleRM_data_pt = save_dir / "1_style_removed_images"
	noisy_latent_pt = save_dir / "2_fdp_latents"

	# infer_dataset = MRI.DWITHP_t1t2(data_pt, lb_val_combined, lb_val_tar, image_min=image_min) # for .npy data
	infer_dataset = MRI.MRI_nii_Dataset_3D(data_pt, lb_inference, image_min=-1.0) # for .nii.gz data
	infer_loader = DataLoader(
		infer_dataset, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True, drop_last=False
	)

	if not styleRM_data_pt.exists():
		style_removal(
			infer_loader,
			unet,
			scheduler_ddim,
			inferer,
			save_dir,
			condition_on,
			grad_cond_type,
			ema_mean,
			ema_std,
			image_min,
			num_inference_fdp,
			num_inference_rdp,
			use_amp,
		)
		if apply_scale_images:
			scale_images(styleRM_data_pt, target_image=target_image, image_min=image_min)

	if not noisy_latent_pt.exists():
		# sr_dataset = MRI.MRI_nii_Dataset_3D(styleRM_data_pt, lb_inference, image_min=-1.0, absolute_path=False)
		sr_dataset = StyleRmNpyDataset(styleRM_data_pt, lb_inference)
		sr_loader = DataLoader(
			sr_dataset, batch_size=4, shuffle=False, num_workers=2, persistent_workers=True, drop_last=False
		)
		pre_compute_latent(
			sr_loader,
			unet,
			scheduler_ddim,
			inferer,
			save_dir,
			condition_on,
			grad_cond_type,
			ema_mean,
			ema_std,
			num_inference_fdp,
			use_amp,
		)

	latent_dataset = LatentDataset(styleRM_data_pt, noisy_latent_pt, lb_inference)
	latent_loader = DataLoader(
		latent_dataset, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True, drop_last=False
	)

	unet.load_state_dict(stage2_ckp["model"])
	unet.eval()
	print("Stage 2 model loaded.")

	wd = CLIP_val(
		unet,
		latent_loader,
		target_image,
		target_site,
		transform,
		brain_mask,
		scheduler_ddim,
		inferer,
		ema_mean,
		ema_std,
		image_min,
		num_inference_rdp,
		save_dir,
		save_outputs=True,
		use_amp=use_amp,
	)

	with open(save_dir / "inference_metrics.txt", "w") as f:
		f.write(f"WD={wd}\n")
