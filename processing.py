from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os
from torchvision.models import vit_l_16, ViT_L_16_Weights
import torch
from feature_extraction import ImageDataset


images_path = os.path.join("images", "data")
head_path = os.path.join("checkpoints", "head.pth")

weights = ViT_L_16_Weights.IMAGENET1K_V1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit = vit_l_16(weights=weights).to(device)
vit.eval()

head = torch.nn.Linear(in_features=1000, out_features=2, device=device)
head.load_state_dict(torch.load(head_path))
head.eval()

transform = weights.transforms()

dataset = ImageDataset(images_path, transform)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

saved = 0
predictions = torch.tensor([]).to(device)
for images in loader:
    images = images.to(device)
    with torch.no_grad():
        features = vit(images)
        output = head(features).argmax(dim=1)
        predictions = torch.cat((predictions, output))