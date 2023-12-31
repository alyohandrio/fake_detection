from torch.utils.data import DataLoader
import os
from torchvision.models import vit_l_16, ViT_L_16_Weights
import torch
from utils import ImageDataset


def predict_fakes(images_path, head_path=None):
    if head_path is None:
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
    return predictions, dataset.get_names()

