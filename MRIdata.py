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

License: MIT License (see LICENSE file for details)
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




class DWITHP_t1t2(MRIdata_3d_t1t2):
	"""Dataset for DWI THP T1 and T2 images"""
	def __init__(self, image_dir='', src_combined_lb=None, tar_t1_lb=None, tar_t2_lb=None, image_min=0.0):
		print(f'number of src_combined_lb: {len(src_combined_lb) if src_combined_lb is not None else "None"}')
		print( f'number of tar_t1_lb: {len(tar_t1_lb) if tar_t1_lb is not None else "None"}')
		print( f'number of tar_t2_lb: {len(tar_t2_lb) if tar_t2_lb is not None else "None"}')
		super().__init__(image_dir, src_combined_lb, tar_t1_lb, tar_t2_lb, image_min=image_min)