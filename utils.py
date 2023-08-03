import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.ToTensor()
        self.names = [name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))]

    def __len__(self):
        return len(self.names)

    def get_names(self):
        return self.names

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.names[idx])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

class FeaturesDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.real_features = torch.load(os.path.join(root, "0", "result.pt"))
        self.fake_features = torch.load(os.path.join(root, "1", "result.pt"))

    def __len__(self):
        return len(self.real_features) + len(self.fake_features)

    def __getitem__(self, idx):
        if idx < len(self.real_features):
            return self.real_features[idx], 0
        else:
            return self.fake_features[-len(self.real_features) + idx], 1


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
def validation_epoch(model, criterion, val_loader):
    val_loss, val_accuracy = 0.0, 0.0
    model.eval()
    device = next(model.parameters()).device
    for features, labels in val_loader:
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

