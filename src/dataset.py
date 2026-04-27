# src/dataset.py

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
from src.config import IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None, sample_frac=1.0, image_path_col="path"):
        self.df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        self.transform = transform
        self.image_path_col = image_path_col

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

        image = Image.open(PROJECT_ROOT / row[self.image_path_col]).convert("RGB")
        label = self.label_map[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])