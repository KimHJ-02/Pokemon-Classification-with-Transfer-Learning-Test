import os
from PIL import Image
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.classes = []
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)
        # scan
        for cls in sorted(os.listdir(self.root_dir)):
            cls_path = os.path.join(self.root_dir, cls)
            if os.path.isdir(cls_path):
                self.classes.append(cls)
                for fn in os.listdir(cls_path):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cls_path, fn), cls))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label


def load_class_names(root_dir):
    train_dir = os.path.join(root_dir, 'train')
    if not os.path.isdir(train_dir):
        return []
    return sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
