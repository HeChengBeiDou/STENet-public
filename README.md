# STENet
This is the official code repository for "STENet: A Multi-task Segmentation Network with Spatial Topology Enhancement for Fetal Echocardiography
".

## Abstract
Congenital heart disease (CHD) poses substantial health risks to developing fetuses. Echocardiography with the apical four-chamber (A4C) view is essential for early CHD diagnosis, as accurate fetal cardiac segmentation provides critical anatomical details on cardiac structures, aiding in function assessment. However, fetal echocardiography encounters two main challenges: variable cardiac chamber position in the A4C view, complicating model training, and the risk of chamber confusion, which together lead to segmentation inaccuracies and diagnostic errors. To address these issues, we propose the multi-task Spatial Topology Enhancement Network (STENet) for fetal cardiac segmentation. STENet combines a multi-task framework to predict segmentation and class centroids, utilizing a spatial topological prior (STP) module to stabilize features against chamber position variability and a spatial information fusion (SIF) module to improve chamber differentiation. Experimental results on a fetal A4C echocardiography dataset show STENet’s superior performance over existing methods, with notable improvements in the Dice coefficient, IoU, and HD95, which are
0.8998, 0.8322, and 1.9565. Our results highlight STENet’s effectiveness in
overcoming key challenges in fetal cardiac segmentation, thereby enabling more
reliable anatomical analysis for clinical assessment.


## 1. Train the VM-UNet
```bash
cd code
python train_stage.py
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMUnet] for the open-source codes.
