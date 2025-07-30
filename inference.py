import torch
from sfcn import SFCN
import os
from read_xlsx_test import *
from fdataset import FetalBrainDataset
from tqdm import tqdm

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

def test(device, file, model, dataloader, split, output_dir):

    assert split == 'test' or "GMH" or "VM" or "SEC"
    if split == "test":
        id_age_map = create_id_to_age_lookup(file) 
        start = ''
    elif split == "GMH":
        id_age_map = create_id_to_age_lookup(file) 
        start = 'gmh_'
    elif split == "VM":
        id_age_map = create_id_to_age_lookup(file) 
        start = 'vm_'
    else:
        id_age_map = create_id_to_age_lookup(file) 
        start = 'sec_'
    output_dir = Path(output_dir) / split
    age_record_path = output_dir / "age_records.csv"  
    os.makedirs(output_dir, exist_ok=True)  
    with open(age_record_path, 'w') as f:  
        f.write("brain_id,original_age,predicted_age\n")  
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            brain, filenames = batch
            brain = brain.to(device)
            ages = [id_age_map.get(start + fname.split('_')[0], float('nan')) for fname in filenames]   
            age = torch.tensor(ages, dtype=torch.float32).unsqueeze(-1).to(device)
            brain_id = filenames[0].split('_')[0] 
            predicted_age = model(brain)
            predicted_age = predicted_age[0].cpu().numpy().squeeze()
            original_age = age.cpu().numpy().squeeze()

            with open(age_record_path, 'a') as f:  
                f.write(f"{brain_id},{original_age},{predicted_age}\n")  

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda')  
    model = SFCN(select_patch=True).to(device)

    checkpoint_path = 'best_model.pt'  
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['best_val_loss']
        print(f'Checkpoint loaded: {checkpoint_path}, epoch: {epoch}, loss: {loss}')
    else:
        print(f'Checkpoint not found: {checkpoint_path}')

    
    test_datapath1 = "/path/to/normal"
    test_dataset1 = FetalBrainDataset(test_datapath1, split = 'test')
    print(f"Original train_dataset size: {len(test_dataset1)}")
    test_loader1 = torch.utils.data.DataLoader(test_dataset1,batch_size=1,shuffle=False,drop_last=False,num_workers=6) 

    
    test_datapath2 = "/path/to/gmh-ivh"
    test_dataset2 = FetalBrainDataset(test_datapath2, split = 'test')
    print(f"Original train_dataset size: {len(test_dataset2)}")
    test_loader2 = torch.utils.data.DataLoader(test_dataset2,batch_size=1,shuffle=False,drop_last=False,num_workers=6) 
    
    test_datapath3 = "/path/to/vm"
    test_dataset3 = FetalBrainDataset(test_datapath3, split = 'test')
    print(f"Original train_dataset size: {len(test_dataset3)}")
    test_loader3 = torch.utils.data.DataLoader(test_dataset3,batch_size=1,shuffle=False,drop_last=False,num_workers=6) 
    
    test_datapath4 = "/path/to/sec"
    test_dataset4 = FetalBrainDataset(test_datapath4, split = 'test')
    print(f"Original train_dataset size: {len(test_dataset4)}")
    test_loader4 = torch.utils.data.DataLoader(test_dataset4,batch_size=1,shuffle=False,drop_last=False,num_workers=6) 

    # Files below represent a table that includes ids and correspend ages, please use your own instead.
    test(device, file1, model, test_loader1, split = "test", output_dir=output_dir)
    test(device, file2, model, test_loader2, split = "GMH", output_dir=output_dir)
    test(device, file3, model, test_loader3, split = "VM", output_dir=output_dir)
    test(device, file4, model, test_loader4, split = "SEC", output_dir=output_dir)
