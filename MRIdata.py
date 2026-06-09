#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: MRIdata.py
Description: Dataset class for multi-sequence MRI data loading

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

import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms as tr

from pathlib import Path
import pandas as pd

	
class MRIdata_3d_t1t2(Dataset):
	def __init__(self,image_dir:str,src_combined_lb, tar_t1_lb, tar_t2_lb=None,image_min=0.0): 
		assert Path(image_dir).is_dir(), f'{image_dir} is not a valid directory'
		self.image_min = image_min
		
		self.img_dir = image_dir
		self.src_combined_lb = src_combined_lb
		self.tar_t1_lb = tar_t1_lb
		self.tar_t2_lb = tar_t2_lb
		
		self.length = len(self.src_combined_lb)
		self.transform = tr.Compose([
					tr.ToTensor(),
					tr.CenterSpatialCrop(roi_size=(144, 184, 184)),
				])
		self.load_target = True if (self.tar_t1_lb is not None) or (self.tar_t2_lb is not None) else False

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		img_path_base = Path(self.img_dir)
		fn  = str(self.src_combined_lb.iloc[idx,0])
		img_path_full = str(next(img_path_base.glob(f'*{fn}.npy')))
		img_volume = np.load(img_path_full).astype(np.float32)

		img_volume = torch.from_numpy(img_volume)
		if len(img_volume.shape) != 4:
			img_volume = img_volume.unsqueeze(0)

		if 'T2' in fn:
			random_idx = np.random.randint(0, len(self.tar_t2_lb))
			if self.load_target:
				tar_fn = str(self.tar_t2_lb.iloc[random_idx, 0])  # Randomly select a T2 target image
			class_emb = torch.tensor(1).int()
		else:
			random_idx = np.random.randint(0, len(self.tar_t1_lb))
			if self.load_target:
				tar_fn = str(self.tar_t1_lb.iloc[random_idx, 0]) # Randomly select a T1 target image
			class_emb = torch.tensor(0).int()
		
		if self.load_target:
			tar_img_path_full = str(next(img_path_base.glob(f'*{tar_fn}.npy')))
			tar_img_volume = np.load(tar_img_path_full).astype(np.float32)
			tar_img_volume = torch.from_numpy(tar_img_volume)
			if len(tar_img_volume.shape) != 4:
				tar_img_volume = tar_img_volume.unsqueeze(0)

			img_volume = self.transform(img_volume)
			tar_img_volume = self.transform(tar_img_volume)
		else:
			img_volume = self.transform(img_volume)
			tar_img_volume = img_volume
			tar_fn = ''

		if ('T2' in fn) and not ('T2' in tar_fn):
			raise ValueError(f'T2 image {fn} is not paired with T2 target {tar_fn}')
		elif (not 'T2' in fn) and ('T2' in tar_fn):
			raise ValueError(f'T1 image {fn} is not paired with T1 target {tar_fn}')
		
		if self.image_min == -1.0 and img_volume.min() >= 0.0:
			img_volume = img_volume * 2.0 - 1.0
			tar_img_volume = tar_img_volume * 2.0 - 1.0
		
		example = {'image':img_volume,'fn':fn, 'target': tar_img_volume, 'tar_fn': tar_fn,'class_emb': class_emb}

		return example



class MRI_nii_Dataset_3D(Dataset):
	def __init__(self, data_root_dir: str, df, transform=None, transform_mask=None, image_min=-1.0, absolute_path=True):
		self.data_root_dir = Path(data_root_dir) # paths to the data directories of each dataset
		self.df = df
		self.image_min = image_min
		self.absolute_path = absolute_path
		if (transform is not None) and (transform_mask is not None):
			self.transform = transform
			self.mask_transform = transform_mask
		else:
			self.transform = tr.Compose([
				tr.SpatialPad(spatial_size=(184, 184, 184)),
				# tr.CenterSpatialCrop(roi_size=(184, 184, 184)),
				tr.CenterSpatialCrop(roi_size=(144, 184, 184)),

				tr.ScaleIntensityRangePercentiles(
					lower=0, upper=99.5, b_min=image_min, b_max=1.0, clip=True
				),
				tr.ToTensor(),
			])
			

			self.mask_transform = tr.Compose([
				tr.SpatialPad(spatial_size=(184, 184, 184)),
				# tr.CenterSpatialCrop(roi_size=(184, 184, 184)),
				tr.CenterSpatialCrop(roi_size=(144, 184, 184)),
				tr.ToTensor(),
			])



	def __len__(self):
		return len(self.df)


	def __getitem__(self, idx):
		dataset_root = self.data_root_dir 

		if self.absolute_path:
			scan_path = dataset_root / self.df.iloc[idx]['ImageFilePath']
			mask_path = dataset_root / self.df.iloc[idx]['MaskFilePath']
		else:
			scan_path = dataset_root / f"{Path(self.df.iloc[idx]['ImageFilePath']).name.replace('.nii.gz','.npy')}"
			mask_path = dataset_root / f"{Path(self.df.iloc[idx]['MaskFilePath']).name.replace('.nii.gz','.npy')}"

		group = self.df.iloc[idx]['ResearchGroup']
		subject = self.df.iloc[idx]['SubjectID']
		class_emb = torch.tensor(0).int()
		fn = scan_path.name.replace('.nii.gz','')
	 
		try:
			scan =  torch.from_numpy(nib.load(str(scan_path)).get_fdata(dtype=np.float32))  # Load as float32 for better precision during transforms
			# org_scan = scan.clone()  # Keep original for debugging
		except FileNotFoundError as e:
			print(f"[NIFTI-LOAD-ERROR] Missing file: {scan_path}")
			raise e
		except (ImageFileError, OSError, EOFError, ValueError) as e:
			print(f"[NIFTI-LOAD-ERROR] Not loadable by nibabel (expected .nii/.nii.gz): {scan_path}")
			print(f"[NIFTI-LOAD-ERROR] {type(e).__name__}: {e}")
			raise e       
	   
		# print(f"Original scan shape: {scan.shape}")
		# print(f"Original mean intensity: {scan.mean():.4f}, std intensity: {scan.std():.4f}, min intensity: {scan.min():.4f}, max intensity: {scan.max():.4f}")
		scan = self.transform(scan.unsqueeze(0)).clamp(self.image_min,1).float()  # Add channel dimension for MONAI transforms
		# print(f"Transformed scan shape: {scan.shape}")
		# print(f"Transformed mean intensity: {scan.mean():.4f}, std intensity: {scan.std():.4f}, min intensity: {scan.min():.4f}, max intensity: {scan.max():.4f}")

		try:
			mask_file = torch.from_numpy(nib.load(str(mask_path)).get_fdata(dtype=np.float32))
		except FileNotFoundError as e:
			print(f"[NIFTI-LOAD-ERROR] Missing file: {mask_path}")
			raise e
		except (ImageFileError, OSError, EOFError, ValueError) as e:
			print(f"[NIFTI-LOAD-ERROR] Not loadable by nibabel (expected .nii/.nii.gz): {mask_path}")
			print(f"[NIFTI-LOAD-ERROR] {type(e).__name__}: {e}")
			raise e

		mask_file = self.mask_transform(mask_file.unsqueeze(0)).float()  # Add channel dimension for MONAI transforms



		# return {'image': scan, 'brain_mask': mask_file,'org_scan': org_scan}  # Return original scan for debugging
		out_dict = {
			'image': scan, 'brain_mask': mask_file, 
			'ImageUID': self.df.iloc[idx]['ImageUID'], 
			'group': group,
			'SubjectID': subject,
			'class_emb': class_emb,
			'fn': fn
		}


		return out_dict



class DWITHP_t1t2(MRIdata_3d_t1t2):
	"""Dataset for DWI THP T1 and T2 images"""
	def __init__(self, image_dir='', src_combined_lb=None, tar_t1_lb=None, tar_t2_lb=None, image_min=0.0):
		print(f'number of src_combined_lb: {len(src_combined_lb) if src_combined_lb is not None else "None"}')
		print( f'number of tar_t1_lb: {len(tar_t1_lb) if tar_t1_lb is not None else "None"}')
		print( f'number of tar_t2_lb: {len(tar_t2_lb) if tar_t2_lb is not None else "None"}')
		super().__init__(image_dir, src_combined_lb, tar_t1_lb, tar_t2_lb, image_min=image_min)