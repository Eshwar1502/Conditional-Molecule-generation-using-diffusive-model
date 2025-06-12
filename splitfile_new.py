import pickle
import torch
from datasets import ZINC250k_7props

dataset = ZINC250k_7props(root='data')
indices = torch.randperm(len(dataset))
splits = {
    'train_idx': indices[:int(0.8*len(dataset))].tolist(),
    'valid_idx': indices[int(0.8*len(dataset)):].tolist()
}

with open('data/raw/splits_7props.pkl', 'wb') as f:
    pickle.dump(splits, f)