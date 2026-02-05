import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .eurosat import EuroSATDataset
from .gtsrb import GTSRBDataLoader
from torch.utils.data import random_split


def get_dataloader(config, dev_mode=False, dev_ratio=0.1):
    """
    Get appropriate dataloader based on configuration.

    Args:
        config: Configuration dictionary with dataset and model details

    Returns:
        data_loader: A data loader object
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
    """
    dataset_name = config["dataset"]["name"].lower()
    modality = config["dataset"]["modality"].lower()
    batch_size = config["dataset"]["batch_size"]
    image_size = config["dataset"].get("image_size", 224)
    if modality == "image":
        return _get_image_dataloader(config, dataset_name, batch_size, image_size, dev_mode, dev_ratio)
    elif modality == "text":
        return _get_text_dataloader(config, dataset_name, batch_size, dev_mode, dev_ratio)
    else:
        raise ValueError("Unsupported modality type. Choose 'text' or 'image'.")


def _get_image_dataloader(config, dataset_name, batch_size, image_size, dev_mode=False, dev_ratio=0.1):
    """Handle image datasets"""

    if dataset_name == "eurosat":
        root_path = "/oscar/data/sbach/bats/datasets/eurosat/EuroSAT_RGB"
        data_loader = EuroSATDataset(root_path, image_size=image_size)
        num_classes = 10
        train_dataloader = DataLoader(
            data_loader.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([data_loader.clip_transform(Image.open(img).convert("RGB")) for img, _ in batch]),
                torch.tensor([data_loader.class_to_idx[lbl] for _, lbl in batch])
            ),
        )
        val_dataloader = DataLoader(
            data_loader.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([data_loader.clip_transform(Image.open(img).convert("RGB")) for img, _ in batch]),
                torch.tensor([data_loader.class_to_idx[lbl] for _, lbl in batch])
            ),
        )
        non_shuffle_train_dataloader = DataLoader(
                    data_loader.train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda batch: (
                        torch.stack([data_loader.clip_transform(Image.open(img).convert("RGB")) for img, _ in batch]),
                        torch.tensor([data_loader.class_to_idx[lbl] for _, lbl in batch])
                    ),
                )
        test_dataloader = DataLoader(
            data_loader.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([data_loader.clip_transform(Image.open(img).convert("RGB")) for img, _ in batch]),
                torch.tensor([data_loader.class_to_idx[lbl] for _, lbl in batch])
            ),
        )
    elif dataset_name == "gtsrb":
        # TODO: Update these paths to match your GTSRB dataset location
        train_root = "./datasets/gtsrb/GTSRB/Training"  # TODO: Set your GTSRB training path
        test_annotations = "./datasets/gtsrb/GT-final_test.csv"  # TODO: Set your GTSRB test annotations path
        test_img_dir = "./datasets/gtsrb/GTSRB/Final_Test/Images"  # TODO: Set your GTSRB test images path
        data_loader = GTSRBDataLoader(train_root, test_annotations, test_img_dir, image_size = image_size)
        num_classes = 43
        full_train = data_loader.train_dataset
        test_dataset = data_loader.test_dataset

        if dev_mode:
            val_len = int(len(full_train) * dev_ratio)
            train_len = len(full_train) - val_len
            train_dataset, val_dataset = random_split(
                full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
        else:
            train_dataset = full_train
            val_dataset = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )
        val_dataloader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([img for img, _ in batch]),
                    torch.tensor([int(lbl) for _, lbl in batch])
                )
            )
            if val_dataset else None
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )
        non_shuffle_train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )
    elif dataset_name == "clevr_count":
        from .clevr_count import CLEVRCountDataLoader
        # Initialize the dataloader
        data_loader = CLEVRCountDataLoader(image_size = image_size)
        num_classes = data_loader.num_classes  # Should be 8 (0-7 objects)
        full_train = data_loader.train_dataset
        test_dataset = data_loader.test_dataset

        if dev_mode:
            val_len = int(len(full_train) * dev_ratio)
            train_len = len(full_train) - val_len
            train_dataset, val_dataset = random_split(
                full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
        else:
            train_dataset = full_train
            val_dataset = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )

        val_dataloader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([img for img, _ in batch]),
                    torch.tensor([int(lbl) for _, lbl in batch])
                )
            )
            if val_dataset else None
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )

        non_shuffle_train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: (
                torch.stack([img for img, _ in batch]),
                torch.tensor([int(lbl) for _, lbl in batch])
            )
        )
    else:
        raise ValueError(f"Image dataset '{dataset_name}' not supported.")
    return num_classes, train_dataloader, val_dataloader, test_dataloader, non_shuffle_train_dataloader



def _get_text_dataloader(config, dataset_name, batch_size, dev_mode=False, dev_ratio=0.1):
    """Handle text datasets"""
    # Dataset-specific settings
    dataset_config = {
        # Original datasets OLD!!
        "agnews": {"source": "ag_news", "num_classes": 4, "text_column": "text", "test_split": "test"},
        "amazon": {"source": "amazon_polarity", "num_classes": 2, "text_column": "text", "test_split": "test"},
        "sst": {"source": "stanfordnlp/sst2", "num_classes": 2, "text_column": "sentence",
                "test_split": "validation"},
        "imdb": {"source": "stanfordnlp/imdb", "num_classes": 2, "text_column": "text", "test_split": "test"},

        # GLUE datasets
        "cola": {"source": "glue", "name": "cola", "num_classes": 2, "text_column": "sentence",
                 "test_split": "validation"},
        "mnli": {"source": "glue", "name": "mnli", "num_classes": 3, "text_column": "premise",
                 "second_text_column": "hypothesis", "test_split": "validation_matched"},
        "mnli_matched": {"source": "glue", "name": "mnli", "num_classes": 3, "text_column": "premise",
                         "second_text_column": "hypothesis", "test_split": "validation_matched"},
        "mnli_mismatched": {"source": "glue", "name": "mnli", "num_classes": 3, "text_column": "premise",
                            "second_text_column": "hypothesis", "test_split": "validation_mismatched"},
        "mrpc": {"source": "glue", "name": "mrpc", "num_classes": 2, "text_column": "sentence1",
                 "second_text_column": "sentence2", "test_split": "validation"},
        "qnli": {"source": "glue", "name": "qnli", "num_classes": 2, "text_column": "question",
                 "second_text_column": "sentence", "test_split": "validation"},
        "qqp": {"source": "glue", "name": "qqp", "num_classes": 2, "text_column": "question1",
                "second_text_column": "question2", "test_split": "validation"},
        "rte": {"source": "glue", "name": "rte", "num_classes": 2, "text_column": "sentence1",
                "second_text_column": "sentence2", "test_split": "validation"},
        "sst2": {"source": "glue", "name": "sst2", "num_classes": 2, "text_column": "sentence",
                 "test_split": "validation"},
        "stsb": {"source": "glue", "name": "stsb", "num_classes": 1, "text_column": "sentence1",
                 "second_text_column": "sentence2", "test_split": "validation", "is_regression": True},
        "wnli": {"source": "glue", "name": "wnli", "num_classes": 2, "text_column": "sentence1",
                 "second_text_column": "sentence2", "test_split": "validation"}
    }

    if dataset_name not in dataset_config:
        raise ValueError(f"Text dataset '{dataset_name}' not supported.")

    # Get dataset settings
    ds_config = dataset_config[dataset_name]

    # Load dataset and tokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if ds_config["source"] == "glue":
        dataset = load_dataset(ds_config["source"], ds_config["name"])
    else:
        dataset = load_dataset(ds_config["source"])

    text_column = ds_config["text_column"]
    test_split = ds_config["test_split"]
    has_second_text = "second_text_column" in ds_config
    is_regression = ds_config.get("is_regression", False)

    # Define tokenization function
    def tokenize_batch(batch):
        if has_second_text:
            # For sentence pair tasks
            second_text_column = ds_config["second_text_column"]
            return tokenizer(
                batch[text_column],
                batch[second_text_column],
                padding="max_length",
                truncation=True,
                max_length=config["dataset"]["max_length"]
            )
        else:
            # For single sentence tasks
            return tokenizer(
                batch[text_column],
                padding="max_length",
                truncation=True,
                max_length=config["dataset"]["max_length"]
            )

    # Tokenize datasets
    remove_columns = [text_column]
    if has_second_text:
        remove_columns.append(ds_config["second_text_column"])

    # train_dataset = dataset["train"].map(tokenize_batch, remove_columns=remove_columns, batched=True)
    # test_dataset = dataset[test_split].map(tokenize_batch, remove_columns=remove_columns, batched=True)
    full_train = dataset["train"].map(tokenize_batch, remove_columns=remove_columns, batched=True)
    test_dataset = dataset[test_split].map(tokenize_batch, remove_columns=remove_columns, batched=True)

    if dev_mode:
       val_len = int(len(full_train) * dev_ratio)
       train_len = len(full_train) - val_len
       train_dataset, val_dataset = torch.utils.data.random_split(
           full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42)
       )
    else:
       train_dataset = full_train
       val_dataset = None    
    def collate_fn(batch):
        input_ids = torch.stack(
            [torch.tensor(item["input_ids"]) if isinstance(item["input_ids"], list) else item["input_ids"] for item
             in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) if isinstance(item["attention_mask"],
                                                                                         list) else item[
            "attention_mask"] for item in batch])

        has_token_type_ids = "token_type_ids" in batch[0]

        if is_regression:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
        else:
            labels = torch.tensor([item["label"] for item in batch])

        if has_token_type_ids:
            token_type_ids = torch.stack([torch.tensor(item["token_type_ids"]) if isinstance(item["token_type_ids"],
                                                                                             list) else item[
                "token_type_ids"] for item in batch])
            return input_ids, attention_mask, token_type_ids, labels
        else:
            return input_ids, attention_mask, labels

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
       if val_dataset else None
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Prepare DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    non_shuffle_train_dataloader = DataLoader(
         train_dataset,
         batch_size=batch_size,
         shuffle=False,
         collate_fn=collate_fn
    )


    return ds_config["num_classes"], train_dataloader, val_dataloader, test_dataloader, non_shuffle_train_dataloader
