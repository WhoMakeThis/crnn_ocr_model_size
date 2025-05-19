import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # CTC blank = 0
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = ""  # for CTC blank

MAX_LABEL_LEN = 5

def text_to_labels(text):
    return [CHAR2IDX[c] for c in text]

def labels_to_text(labels):
    return ''.join([IDX2CHAR[idx] for idx in labels if idx != 0])

class CaptchaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = sorted([
            os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((50, 150)),  # ✅ height=50 고정
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        filename = os.path.basename(img_path)
        label_text = os.path.splitext(filename)[0][:MAX_LABEL_LEN]
        label = text_to_labels(label_text)

        return img, label, len(label)

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths
