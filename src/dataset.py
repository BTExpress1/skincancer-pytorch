# src/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.config import IMAGE_SIZE


class SkinCancerDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, sample_frac=1.0):
        self.df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.label_map = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        label = self.label_map[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])