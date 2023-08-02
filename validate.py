from utils import train, FeaturesDataset
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str)
parser.add_argument("--validation", type=str)
args = parser.parse_args()

max_iterations = 100000
total_epochs = {}
best_loss = {}
horizon = 100
lrs = np.logspace(-1, -4, 20, base=10)

train_dataset = FeaturesDataset(args.train)
val_dataset = FeaturesDataset(args.validation)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for lr in tqdm(lrs):
    model = torch.nn.Linear(1000, 2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_epochs[lr] = max_iterations
    val_losses = []
    for i in range(max_iterations):
        _, val_loss, _, _ = train(model, optimizer, criterion, 1, train_loader, val_loader)
        val_loss = val_loss[0]
        if i >= horizon and val_losses[-horizon] < val_loss:
            idx = i - horizon
            total_epochs[lr] = 1 + idx + val_losses[idx:].index(min(val_losses[idx:]))
            break
        val_losses += [val_loss]
    best_loss[lr] = val_losses[total_epochs[lr] - 1]

best_lr = sorted(best_loss.items(), key=lambda x: x[1])[0][0]
print(f"Best learning rate: {best_lr}, {total_epochs[best_lr]} epochs")

