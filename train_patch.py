from sfcn import SFCN
from monai.networks.nets import ResNet, ResNetBottleneck
import torch
from fdataset import FetalBrainDataset
from pathlib import Path
import random
import warnings
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F
import numpy as np
import os  
from torch.utils.data import Dataset, Subset  
import nibabel as nib  
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd


def create_id_to_age_lookup(file_path, id_column='ID', age_column='Age'):  
    df = pd.read_excel(file_path, engine='openpyxl')  
    id_age_dict = df.set_index(id_column)[age_column].to_dict()  
    result_dict = {}  
    for id_key, age_value in id_age_dict.items():  
        input_str = str(age_value)  
        if "+" in input_str:  
            parts = input_str.split("+")  
            result = int(parts[0]) * 7 + int(parts[1])  
        else:  
            result = int(input_str) * 7  
        result_dict[id_key] = result  
    return result_dict  
file = '/path/to/age'  
id_age_map = create_id_to_age_lookup(file) 

def loss_f2(age, recon_age):
    return torch.abs(age - recon_age).mean()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        brain, filenames = batch
        brain = brain.to(device)
        optimizer.zero_grad()
        ages = [id_age_map.get(fname.split('_')[0], float('nan')) for fname in filenames]   
        age = torch.tensor(ages, dtype=torch.float32).to(device)
        predicted_age = model(brain)[0].squeeze()
        loss = loss_f2(age, predicted_age)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    len_loss = len(train_loader)
    return total_loss / len_loss


def val(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            brain, filenames = batch
            brain = brain.to(device)
            ages = [id_age_map.get('fb' + fname.split('_')[0], float('nan')) for fname in filenames]   
            age = torch.tensor(ages, dtype=torch.float32).to(device)
            predicted_age = model(brain)[0].squeeze()
            loss = loss_f2(age, predicted_age)
            total_loss += loss.item()
        len_loss = len(val_loader)
    return total_loss / len_loss
if __name__ == "__main__":
    gpu_index = 0
    device = torch.device(f'cuda:{gpu_index}')  
    model = SFCN(select_patch=True).to(device)
    lr = 0.0001
    max_epochs = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001)
    train_datapath = "/data/birth/lmx/work/Class_projects/course5/work/fetal/Data/new_age_data/train"
    train_dataset = FetalBrainDataset(train_datapath, split = 'test')
    print(f"Original train_dataset size: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True,num_workers=6) 

    val_datapath = "/data/birth/lmx/work/Class_projects/course5/work/fetal/Data/new_age_data/val"
    val_dataset = FetalBrainDataset(val_datapath, split = 'test')
    print(f"Original val_dataset size: {len(val_dataset)}")
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False,drop_last=True,num_workers=6) 

    progress_bar = tqdm(range(1, max_epochs + 1), desc="Training Progress")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

for epoch in progress_bar:
    train_loss = train(model, train_dataloader, optimizer, device)
    train_losses.append(train_loss)
    
    val_loss = val(model, val_dataloader, device)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch}/{1000}]")  
    print(f"Loss : Train: {train_loss:.8f}, Val: {val_loss:.8f}")  

    output_folder = "/your/output/folder"

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join('/your/output/folder', 'best_model.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss}, checkpoint_path)

    if epoch % 25 == 0:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        epochs = range(4, len(train_losses) + 1)  
        plt.figure(figsize=(10, 5))  
        plt.plot(epochs, train_losses[3:], color='green', label='Train Loss Age')  
        plt.plot(epochs, val_losses[3:], color='orange', label='Val Loss Age')  
        plt.title('Age Loss')  
        plt.xlabel('Epoch')  
        plt.ylabel('Loss')  
        plt.legend()  
        plt.grid(True)  
        plt.tight_layout()  
        age_loss_path = os.path.join(output_folder, f"age_loss_{current_time}.png")  
        plt.savefig(age_loss_path, bbox_inches='tight', pad_inches=0)  
        plt.close()