import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EuroSATDataset(Dataset):
    def __init__(self, root_dir, image_size = 224,seed=82, transform=None):
        self.root_dir = Path(root_dir)
        self.clip_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]) if transform is None else transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Read class folders
        rng = random.Random(seed)  
        class_folders = sorted([f for f in self.root_dir.iterdir() if f.is_dir()])

        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            # Collect image paths
            image_paths = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
            rng.shuffle(image_paths)  
            # Split 8:1:1
            n_total = len(image_paths)
            n_train = int(n_total * 0.8)
            n_valid = int(n_total * 0.1)
            n_test = n_total - n_train - n_valid

            self.samples.extend([(img, class_name, "train") for img in image_paths[:n_train]])
            self.samples.extend([(img, class_name, "valid") for img in image_paths[n_train:n_train + n_valid]])
            self.samples.extend([(img, class_name, "test") for img in image_paths[n_train + n_valid:]])

        self.train_dataset = [(img, lbl) for img, lbl, split in self.samples if split == "train"]
        self.val_dataset = [(img, lbl) for img, lbl, split in self.samples if split == "valid"]
        self.test_dataset = [(img, lbl) for img, lbl, split in self.samples if split == "test"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx][:2]
        image = Image.open(img_path).convert("RGB")
        image = self.clip_transform(image)
        return image, class_name  # Returning class_name as text
