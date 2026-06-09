# Unified Multi-Site Multi-Sequence Brain MRI Harmonization (MMH) Enriched by Biomedical Semantic Style
This is the official code repository for the paper ["_Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style_"](https://arxiv.org/abs/2601.08193) (under review).


**Abstract:**
Aggregating multi-site brain MRI data can enhance deep learning model training, but also introduces non-biological heterogeneity caused by site-specific variations (e.g., differences in scanner vendors, acquisition parameters, and imaging protocols) that can undermine generalizability. Recent retrospective MRI harmonization seeks to reduce such site effects by standardizing image style (e.g., intensity, contrast, noise patterns) while preserving anatomical content. However, existing methods often rely on limited paired traveling-subject data or fail to effectively disentangle style from anatomy. Furthermore, most current approaches address only single-sequence harmonization, restricting their use in real-world settings where multi-sequence MRI is routinely acquired. To this end, we introduce MMH, **a unified framework for multi-site multi-sequence brain MRI harmonization** that leverages biomedical semantic priors for sequence-aware style alignment. MMH operates in two stages: (1) a diffusion-based global harmonizer that maps MR images to a sequence-specific unified domain using style-agnostic gradient conditioning, and (2) a target-specific fine-tuner that adapts globally aligned images to desired target domains. A tri-planar attention BiomedCLIP encoder aggregates multi-view embeddings to characterize volumetric style information, allowing explicit disentanglement of image styles from anatomy without requiring paired data. Evaluations on 4,163 T1- and T2-weighted MRIs demonstrate MMH’s superior harmonization over state-of-the-art methods in image feature clustering, voxel-level comparison, tissue segmentation, and downstream age and site classification.

![alt text](assets/pipeline_figure.png)

---

## Installation & Environment
### Conda Setup
```bash
conda env create -f environment.yml
conda activate MMH_env
```

---

## Data Preparation
### Preprocessing
The model expects all 3D brain MRIs to undergo minimal preprocessing. The required steps include:

* **FOV Reorientation**
* **Neck Cropping**
* **Bias-Field Correction**
* **Skull Stripping** (Optional)
* **Linear Registration** (to the MNI-152 template)

> 💡 **Preprocessing Pipelines:** For comprehensive, ready-to-use pipelines that handle these steps, check out:
> * FreeSurfer [`recon-all`](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)
> * FSL [`fsl_anat`](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/fsl_anat.html)

All preprocessed NIfTI images should be further center-cropped to [144, 184, 184], scaled to [-1, 1], and converted to NumPy (.npy) format for model trianing.

### Data Organization
The default data loader in `MRIdata.py` is configured to expect the following directory structure:

```bash
dataset_root/
  ├── image_dir/             # Image directory
  │   ├── 00001_S1_T1.npy   # Subject 1, Site 1, T1 scan
  │   ├── 00001_S1_T2.npy   # Subject 1, Site 1, T2 scan
  │   ├── 00001_S2_T1.npy   # Subject 1, Site 2, T1 scan
  │   ├── 00001_S2_T2.npy   # Subject 1, Site 2, T2 scan
  │   ├── 00002_S3_T1.npy  # Subject 2, Site 3, T1 scan
  │   ├── 00002_S3_T2.npy  # Subject 2, Site 3, T2 scan
  │   └── ...  # More subjects scans
  ├── train_labels.csv # Label file for training data
  ├── val_labels.csv # Label file for validation data
  └── test_labels.csv # Label file for test data
```


The `_label.csv` should contains columns like:

| filename    | subject | site | modality |
| ----------- | ------- | ---- | -------- |
| 00001_S1_T1 | 00001   | S1   | T1       |
| 00001_S1_T2 | 00001   | S1   | T2       |
| ...         | ...     | ...  | ...      |


---

## Model Training
### Stage I: Sequence-Specific Global Harmonization
```bash
python train_Stage1.py [options]
```
💡 Run python `train_Stage1.py --help` or view the script directly to see a detailed list of available arguments.
### Stage II: Target-Specific Fine Harmonization
```bash
python train_Stage2.py [options]
```
💡 Run python `train_Stage2.py --help` or view the script directly to see a detailed list of available arguments.

---

## Model Inference
### Stage I: Sequence-Specific Global Harmonization
The Stage I model can be used independently to unify multi-site, multi-sequence MRIs into a sequence-specific unified domain without requiring a specific target.
```bash
python infer_Stage1.py [options]
```
💡 Run python `infer_Stage1.py --help` or view the script directly to see a detailed list of available arguments.

### Stage II: Target-Specific Fine Harmonization 
The Stage II model acts as a target-specific harmonizer to transform an unseen dataset into an existing target domain.
```bash
python infer_Stage2.py [options]
```
💡 Run python `infer_Stage2.py --help` or view the script directly to see a detailed list of available arguments.
***

## ✍️ Citation

If you use this code or our paper in your research, please cite:

```bibtex
@article{wu2026unified,
  title={Unified Multi-Site Multi-Sequence Brain MRI Harmonization Enriched by Biomedical Semantic Style},
  author={Wu, Mengqi and Sun, Yongheng and Wang, Qianqian and Yap, Pew-Thian and Liu, Mingxia},
  journal={arXiv preprint arXiv:2601.08193},
  year={2026}
}

```