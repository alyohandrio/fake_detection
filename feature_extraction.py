from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os
from torchvision.models import vit_l_16, ViT_L_16_Weights
import torch


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


@torch.no_grad()
def process_images(model, loader, path):
    model.eval()
    saved = 0
    for images in loader:
        images = images.to(device)
        features = model(images)
        for feature in features:
            torch.save(feature, os.path.join(path, f"{saved}.pt"))
            saved += 1

weights = ViT_L_16_Weights.IMAGENET1K_V1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit = vit_l_16(weights=weights).to(device)
transform = weights.transforms()

real_path = os.path.join("images", "0")
fake_path = os.path.join("images", "1")
real_dataset = ImageDataset(real_path, transform)
fake_dataset = ImageDataset(fake_path, transform)
real_loader = DataLoader(real_dataset, batch_size=256)
fake_loader = DataLoader(fake_dataset, batch_size=256)

real_path = os.path.join("features", "0")
fake_path = os.path.join("features", "1")
if not os.path.exists(real_path):
    os.makedirs(real_path)
if not os.path.exists(fake_path):
    os.makedirs(fake_path)

process_images(vit, real_loader, real_path)
process_images(vit, fake_loader, fake_path)