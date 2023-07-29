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


NUM_EPOCHS = 30


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=os.path.join("checkpoints", "head.pth"))
args = parser.parse_args()
save_path = args.out
dir_path = os.sep.join(save_path.split(os.sep)[:-1])
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dataset = FeaturesDataset("features")
loader = DataLoader(dataset, batch_size=256, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
head = torch.nn.Linear(in_features=1000, out_features=2, device=device)
optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
head.train()
for _ in tqdm(range(NUM_EPOCHS)):
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
torch.save(head.state_dict(), save_path)

