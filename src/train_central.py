# src/train_central.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.model import create_model
from src.utils import CXRDataset, get_transforms, compute_metrics
import argparse
from tqdm import tqdm
import os
import numpy as np

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(y.numpy().tolist())
    return compute_metrics(ys, preds)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=3, pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_ds = CXRDataset(args.train_csv, args.root_dir, transform=get_transforms(train=True))
    val_ds = CXRDataset(args.val_csv, args.root_dir, transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    best_f1 = 0.0
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = eval_model(model, val_loader, device)
        print(f"Epoch {epoch} loss={loss:.4f} val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(args.out, 'best_model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--out', default='results/central')
    args = parser.parse_args()
    main(args)
