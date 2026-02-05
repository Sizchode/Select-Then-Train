import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset


class CLEVRCountDataLoader:
    def __init__(self, image_size = 224):
        self.clip_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Load the dataset from Hugging Face
        self.dataset = load_dataset("clip-benchmark/wds_vtab-clevr_count_all")

        # CLEVR count has 8 classes (0-7, representing count of objects)
        self.num_classes = 8
        self.class_to_idx = {str(i): i for i in range(self.num_classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Create train and test datasets
        self.train_dataset = CLEVRCountDataset(self.dataset["train"], transform=self.clip_transform)
        self.test_dataset = CLEVRCountDataset(self.dataset["test"], transform=self.clip_transform)

        print("Class to Index Mapping:", self.class_to_idx)
        print(f"Classes represent the count of objects (0-7)")
        print(f"Total training samples: {len(self.train_dataset)}")
        print(f"Total test samples: {len(self.test_dataset)}")


class CLEVRCountDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image and label
        sample = self.dataset[idx]
        image = sample["webp"]  # This should be a PIL Image
        label = sample["cls"]  # This should be the class label (0-7)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, str(label)  # Return label as string for class_to_idx mapping