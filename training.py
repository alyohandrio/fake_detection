import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from utils import FeaturesDataset, train
import argparse


NUM_EPOCHS = 3000

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
optimizer = torch.optim.Adam(head.parameters(), lr=1.4e-5)
criterion = torch.nn.CrossEntropyLoss()
train(head, optimizer, criterion, NUM_EPOCHS, loader)
torch.save(head.state_dict(), save_path)

