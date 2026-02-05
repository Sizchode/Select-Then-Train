import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GTSRBDataLoader:
    def __init__(self, train_root, test_annotations, test_img_dir, image_size = 224):
        self.clip_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = GTSRBTrainDataset(root_dir=train_root, transform=self.clip_transform)
        self.test_dataset = GTSRBTestDataset(annotations_file=test_annotations, img_dir=test_img_dir,
                                             transform=self.clip_transform)

        self.class_to_idx = {str(i): i for i in range(43)}  # GTSRB has 43 classes (0-42)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print("Class to Index Mapping:", self.class_to_idx)


class GTSRBTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Traverse directories for each class (00000 to 00042)
        for class_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                # Load annotations for the class
                annotation_file = os.path.join(class_path, f"GT-{class_dir}.csv")
                annotations = pd.read_csv(annotation_file, sep=';')
                # print(f"Loaded {len(annotations)} samples from {annotation_file}")

                for _, row in annotations.iterrows():
                    img_path = os.path.join(class_path, row['Filename'])
                    label = int(row['ClassId'])
                    self.data.append((img_path, label))
        print(f"Total training samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(label)  # Return label as string for class_to_idx mapping


class GTSRBTestDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file, sep=';')
        self.img_dir = img_dir
        self.transform = transform
        # print(f"Loaded {len(self.annotations)} test samples from {annotations_file}")
        # print("Test dataset column names:", self.annotations.columns.tolist())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Filename'])
        label = int(row['ClassId'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(label)