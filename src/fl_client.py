# src/fl_client.py
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import create_model
from src.utils import CXRDataset, get_transforms, get_parameters, set_parameters, compute_metrics
from typing import Tuple
import argparse
import os

class CXRClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, epochs=1, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # set incoming weights
        set_parameters(self.model, parameters)
        self.model.train()
        for _ in range(self.epochs):
            for x,y in self.train_loader:
                x,y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x,y in self.val_loader:
                x = x.to(self.device)
                logits = self.model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                ys.extend(y.numpy().tolist())
        metrics = compute_metrics(ys, preds)
        # Flower expects (loss, sample_size, {"metric": value})
        # compute loss on val set
        loss_total = 0.0
        criterion = self.criterion
        with torch.no_grad():
            for x,y in self.val_loader:
                x,y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_total += float(criterion(logits,y).item() * x.size(0))
        loss = loss_total / len(self.val_loader.dataset)
        return float(loss), len(self.val_loader.dataset), {"accuracy": metrics['accuracy'], "f1": metrics['f1']}

def start_client(client_id, client_csv, root_dir, server_address="localhost:8080", epochs=1, device='cpu', batch_size=16):
    # build model & dataloaders for this client
    model = create_model(num_classes=3, pretrained=True, device=device)
    train_ds = CXRDataset(client_csv, root_dir, transform=get_transforms(train=True))
    val_ds = CXRDataset(client_csv, root_dir, transform=get_transforms(train=False))  # use local val or split further
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    client = CXRClient(model, train_loader, val_loader, device, epochs=epochs)
    fl.client.start_numpy_client(server_address, client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--client_csv', required=True)
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--server_address', default="localhost:8080")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_client(args.client_id, args.client_csv, args.root_dir, args.server_address, args.epochs, device, args.batch_size)
