

# Patch-based Age Prediction for AD
### 🎯 **Multi-Disease Fetal Brain Anomaly Detection Using Brain Age Prediction: A Patch-based Deep Learning Framework**

## Download pretrained model with the link below
    https://drive.google.com/file/d/1al1h63eVkVSexq1j77lpTFEY_zyYgfzt/view?usp=sharing

## Usage

### 1. Data Preprocessing before training or infernce
    Run preprocess.py with a folder containing *.nii.gz files
### 2. Inference 
    Run python inference_simple.py brain.nii.gz 25 --model best_model.pt
### 3. Train your own age prediction model.
    Prepare an xlsx file with two columns: ID and Age, then run train.py

## Acknowledgements
### We gratefully acknowledge the contributions of the following repos to our work.
1. https://github.com/gift-surg/NiftyMIC
2. https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

