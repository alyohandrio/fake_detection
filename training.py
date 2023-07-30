import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import argparse

class FeaturesDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        real_path = os.path.join(root, "0")
        self.real_names = [name for name in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, name))]
        fake_path = os.path.join(root, "1")
        self.fake_names = [name for name in os.listdir(fake_path) if os.path.isfile(os.path.join(fake_path, name))]

    def __len__(self):
        return len(self.real_names) + len(self.fake_names)

    def __getitem__(self, idx):
        if idx < len(self.real_names):
            item_path = os.path.join(self.root, "0", self.real_names[idx])
            return torch.load(item_path), 0
        else:
            item_path = os.path.join(self.root, "1", self.fake_names[-len(self.real_names) + idx])
            return torch.load(item_path), 1


def training_epoch(model, optimizer, criterion, train_loader):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    device = next(model.parameters()).device
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()
    
    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy

@torch.no_grad()
def validation_epoch(model, optimizer, criterion, val_loader):
    val_loss, val_accuracy = 0.0, 0.0
    model.eval()
    device = next(model.parameters()).device
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        val_loss += loss.item() * features.shape[0]
        val_accuracy += (logits.argmax(dim=1) == labels).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy

def train(model, optimizer, criterion, num_epochs, train_loader, val_loader=None):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for _ in range(num_epochs):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader
        )
        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        if val_loader is not None:
            val_loss, val_accuracy = validation_epoch(
                model, criterion, val_loader
            )
            val_losses += [val_loss]
            val_accuracies += [val_accuracy]
    if val_loader is not None:
        return train_losses, val_losses, train_accuracies, val_accuracies
    else:
        return train_losses, val_losses


NUM_EPOCHS = 30

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=os.path.join("checkpoints", "head.pth"))
parser.add_argument("--features", type=str, default="features")
args = parser.parse_args()
save_path = args.out
dir_path = os.sep.join(save_path.split(os.sep)[:-1])
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dataset = FeaturesDataset(args.features)
loader = DataLoader(dataset, batch_size=256, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
head = torch.nn.Linear(in_features=1000, out_features=2, device=device)
optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train(head, optimizer, criterion, NUM_EPOCHS, loader)
torch.save(head.state_dict(), save_path)

