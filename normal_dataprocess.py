from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
# from block import block_ind

def block_ind(mask, sz_block=32, sz_pad=0):
# find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask)
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    ind_block = ind_block.astype(int)
    return ind_block, ind_brain


def normalise_percentile(volume):  

    v = volume.reshape(-1)  
    v_nonzero = v[v > 0]  # Use only the brain foreground to calculate the quantile  

    p_99 = np.percentile(v_nonzero, 99)

    volume /= p_99

    return volume  

def process_patient(path, target_path):

    # files = sorted(os.listdir(path), key=str.lower)
    # brain_path = os.path.join(path, files[0])  
    # mask_path =  os.path.join(path, os.listdir(path)[0])
    brain_old = nib.load(path).get_fdata()
    # print(brain_old.shape)
    # mask = nib.load(mask_path).get_fdata()
    # brain = brain_old[37:172, 10:199, 27:182]  
    brain = brain_old
    # print(brain.shape)

    ind_block, ind_brain = block_ind(brain_old)
    # print(ind_block)

    brain = normalise_percentile(brain)

    # label_path = os.path.join(path, files[2])
    # labels = nib.load(label_path).get_fdata()

    num_block = ind_block.shape[0]

    patient_dir = Path(target_path)
    patient_dir.mkdir(parents=True, exist_ok=True)
    last_name = patient_dir.parts[-1]
    # print(last_level_name[2:])

    brain_block = [None] * num_block
    # label_block = [None] * num_block
    # (target_path / split).mkdir(parents=True, exist_ok=True)
    for i in range(num_block):

        depth_start = ind_block[i][0]
        depth_end = ind_block[i][1]

        height_start = ind_block[i][2]
        height_end = ind_block[i][3]

        width_start = ind_block[i][4]
        width_end = ind_block[i][5]

        # 获取切片并进行插值 
        brain_block[i] = brain[depth_start:depth_end + 1,
                         height_start:height_end + 1, 
                         width_start:width_end + 1]
       
        
        brain_block[i] = torch.from_numpy(brain_block[i]).float().unsqueeze(dim=0).unsqueeze(dim=0)
        # label_block[i] = torch.from_numpy(label_block[i]).float().unsqueeze(dim=0).unsqueeze(dim=0)
        # combined_block = np.stack([brain_block[i], label_block[i]], axis=0)  
        # np.savez_compressed(patient_dir / f"part{i}", x = combined_block)  

        # np.savez_compressed(patient_dir / f"{last_name[2:]}_part{i}",
        #                         x = brain_block[i])
        
        if i == num_block - 1:
            np.savez_compressed(patient_dir / f"{last_name[0:]}_part{i}_{depth_start}_{depth_end}_{height_start}_{height_end}_{width_start}_{width_end}_{ind_brain[0]}_{ind_brain[1]}_{ind_brain[2]}_{ind_brain[3]}_{ind_brain[4]}_{ind_brain[5]}",
                                x = brain_block[i])
        else:
            np.savez_compressed(patient_dir / f"{last_name[0:]}_part{i}_{depth_start}_{depth_end}_{height_start}_{height_end}_{width_start}_{width_end}",
                                x = brain_block[i])

def preprocess(datapath: Path, targetpath):

    targetpath.mkdir(parents=True, exist_ok=True)
    paths = sorted([f for f in os.listdir(datapath)])

    i = 0
    for source_path in tqdm(paths):
        full_source_path = datapath / source_path  
        # i += 1
        # if(i < 400):
        #     new_targetpath = targetpath / "SEC"
        # elif(i < 440):
        #     new_targetpath = targetpath / "val"
        # else:
        target_path = os.path.join(targetpath, f"{source_path.split('.')[0]}")
        if os.path.exists(target_path):
            continue
        process_patient(full_source_path, target_path)
    
if __name__ == "__main__":

    # datapath = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/new_large_normal/normal/train")
    # targetpath = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/data_split32/train")
    # preprocess(datapath, targetpath)

    # datapath2 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/new_large_normal/normal/val")
    # targetpath2 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/data_split32/val")
    # preprocess(datapath2, targetpath2)

    # datapath3 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/new_large_normal/normal/test")
    # targetpath3 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/data_split32/test")
    # preprocess(datapath3, targetpath3)

    datapath3 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/5_disease/sec_final2")
    targetpath3 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/data_split32/sec")
    preprocess(datapath3, targetpath3)

    datapath4 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/5_disease/lvb_final/test")
    targetpath4 = Path("/data/birth/lmx/lmx_storage/lmx_storage/hyq_storage/data_split32/lvb")
    preprocess(datapath4, targetpath4)

