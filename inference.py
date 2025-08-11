import argparse
import torch
import numpy as np
from sfcn import SFCN
from fdataset import FetalBrainDataset
from tqdm import tqdm

def test(device, model, dataloader, true_age):
    preds = []
    model.eval()
    with torch.no_grad():
        for brain, _ in tqdm(dataloader):
            brain = brain.to(device)
            pred_age = model(brain)
            pred_age = pred_age.cpu().numpy().squeeze()
            preds.append(pred_age)

    preds = np.array(preds)
    meanaad = np.abs(np.mean(preds) - true_age)
    maxaad = np.max(np.abs(preds - true_age))
    
    print("PA of each patch:", np.round(preds, 2))
    print(f"MeanAAD: {meanaad:.2f}")
    print(f"MaxAAD: {maxaad:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_datapath", type=str)
    parser.add_argument("true_age", type=float, help="real GA")
    parser.add_argument("--model", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SFCN(select_patch=True).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded: {args.model}")

    dataset = FetalBrainDataset(args.test_datapath, split='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    test(device, model, dataloader, args.true_age)
