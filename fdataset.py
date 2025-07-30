
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os  
from torch.utils.data import Dataset, Subset    

class FetalBrainDataset(Dataset):  
    def __init__(self, data_dir, split = "train"):  

        self.data_dir = Path(data_dir)  
        self.split = split
        self.data_files = self._collect_data_files(split)  
        self.len = len(self.data_files) 
        self.idx_map = {x: x for x in range(self.len)}
    def _collect_data_files(self, split):  

        data_files = []  

        for patient_dir in sorted(self.data_dir.iterdir()):  
            if patient_dir.is_dir():  
                if(self.split == "train"):
                    patient_files = sorted(patient_dir.glob('*part*.npz'),
                                           key=lambda x: int(x.stem.split('_part')[-1]))
                else: 
                    assert self.split == "test"

                    patient_files = sorted(patient_dir.glob('*_part*.npz'),   
                    key=lambda x: int(x.stem.split('_part')[1].split('_')[0]))   
                data_files.extend(patient_files)  

        return data_files  
    
    def __len__(self):  
        return self.len
    
    def __getitem__(self, idx):  
        idx = self.idx_map[idx]
        data = np.load(self.data_files[idx])

        if(self.split == 'train'):
            data = data[data.files[0]].squeeze(0)  
            # data = torch.from_numpy(data).float()
            return data
        elif(self.split == 'test'):
            # data = process(**data)
            # data = (data[data.files[0]].squeeze(0), data[data.files[1]].squeeze(0))
            data = data[data.files[0]].squeeze(0)
            file_name = os.path.basename(self.data_files[idx])  
            return data, file_name
def random_subset(dataset, fraction = 0.50, seed = 48):  
    total_size = len(dataset)  
    subset_size = int(total_size * fraction)  
    np.random.seed(seed) 
    indices = np.random.choice(  
        total_size,   
        size=subset_size,   
        replace=False
    )  
    subset_dataset = Subset(dataset, indices)  
    return subset_dataset  
