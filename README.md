

# Patch-based Age Prediction for AD
### 🎯 **Multi-Disease Fetal Brain Anomaly Detection Using Brain Age Prediction: A Patch-based Deep Learning Framework**

## Download pretrained model [here](https://drive.google.com/file/d/1al1h63eVkVSexq1j77lpTFEY_zyYgfzt/view?usp=sharing)

## Usage

### 1. Data Preprocessing before training or infernce
Run 
```bash
python preprocess.py
```
with an *nii.gz brain and an output path (modified by yourself).
### 2. Inference 
Run
```bash
    python inference_one_brain.py /path/to/brain_patches 25 --model best_model.pt
```
The path here is the output path of step 1.
### 3. Train your own age prediction model.
Prepare an xlsx file with two columns: ID and Age, then run
```bash
    python train.py
```
## Acknowledgements
### We gratefully acknowledge the contributions of the following repos to our work.
1. https://github.com/gift-surg/NiftyMIC
2. https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

