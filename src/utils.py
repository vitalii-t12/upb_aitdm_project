# src/utils.py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Transforms
def get_transforms(train=True, size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# Simple dataset wrapper assuming CSV with image_path,label
class CXRDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row['image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])  # ensure labels are 0..2
        return img, label

# non-IID split helper (simple Dirichlet or class-skew). Here: class-skew
def create_non_iid_splits(df_csv, root_dir, n_clients=3, dominant_fraction=0.6, out_dir='splits'):
    import os, json
    df = pd.read_csv(df_csv)
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df['label'].unique())
    # assign each client a dominant class (cycle)
    client_splits = {i: [] for i in range(n_clients)}
    for i, cls in enumerate(classes):
        cls_rows = df[df['label'] == cls]
        n = len(cls_rows)
        for client in range(n_clients):
            # if this client is 'dominant' for this class:
            if client == (i % n_clients):
                take = int(n * dominant_fraction)
            else:
                take = int((n * (1 - dominant_fraction)) // (n_clients - 1)) if n_clients>1 else 0
            selected = cls_rows.sample(n=take, replace=False) if take>0 else pd.DataFrame()
            client_splits[client].extend(selected.index.tolist())
    # remaining indices -> distribute round-robin
    used = set(sum(client_splits.values(), []))
    remaining = [i for i in range(len(df)) if i not in used]
    for idx, r in enumerate(remaining):
        client_splits[idx % n_clients].append(r)
    # save csv for each client
    for client, indices in client_splits.items():
        subdf = df.loc[indices].reset_index(drop=True)
        subdf.to_csv(os.path.join(out_dir, f'client_{client}.csv'), index=False)
    return out_dir

# metrics
def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'f1': f1, 'cm': cm}

# FL param helpers: convert state_dict <-> list of np arrays
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
