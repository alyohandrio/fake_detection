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

max_iterations = 50000
total_epochs = {}
best_acc = {}
horizon = 1000
lrs = np.logspace(-2, -5, 20, base=10)

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
    val_accs = []
    for i in range(max_iterations):
        _, _, _, val_acc = train(model, optimizer, criterion, 1, train_loader, val_loader)
        val_acc = val_acc[0]
        if i >= horizon and val_accs[-horizon] > val_acc:
            idx = i - horizon
            total_epochs[lr] = 1 + idx + val_accs[idx:].index(min(val_accs[idx:]))
            break
        val_accs += [val_acc]
    best_acc[lr] = val_accs[total_epochs[lr] - 1]

best_lr = sorted(best_acc.items(), key=lambda x: x[1])[-1][0]
print(f"Best learning rate: {best_lr}, {total_epochs[best_lr]} epochs")

