from datasets import load_dataset, Dataset
import random
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any


class BoolQ:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Q: {question}\nContext:\n{context}\nA:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("super_glue", "boolq")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(
                question=d["question"],
                context=d["passage"]
            )
            answer = "Yes" if d["label"] == 1 else "No"
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("super_glue", "boolq")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set

    def get_dev_set(self, ratio=0.1, return_rest=True):
        """
        Sample a dev set from BoolQ training data, format like test (prompt + target_text + text),
        return remaining training data if requested.
        """
        raw_dataset = load_dataset("super_glue", "boolq")
        train_data = raw_dataset['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))

        train_data = train_data.shuffle(seed=self.seed)
        dev_data_raw = train_data.select(range(sample_size))
        rest_data_raw = train_data.select(range(sample_size, total_samples))

        # Format dev set
        dev_data = []
        for d in dev_data_raw:
            prompt = self.prompt_template.format(
                question=d["question"],
                context=d["passage"]
            )
            answer = "Yes" if d["label"] == 1 else "No"
            text = prompt + answer + '</s>'
            dev_data.append({
                'prompt': prompt,
                'target_text': answer,
                'true_labels': [answer],  # add here
                'text': text
            })

        dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
        print(dev_dataset.column_names) 
        print(f"\n[BoolQ] Dev set sampled: {sample_size}/{total_samples}")
        for i in range(min(3, sample_size)):
            print(f"\n[Sample {i}]")
            print("Prompt:", dev_data[i]['prompt'])
            print("Label:", dev_data[i]['target_text'])

        if return_rest:
            train_dataset_remainder = self.format_prompt(rest_data_raw)
            return dev_dataset, train_dataset_remainder
        else:
            return dev_dataset

    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? A:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("super_glue", "rte")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(
                premise=d["premise"],
                hypothesis=d["hypothesis"]
            )
            answer = "Yes" if d["label"] == 0 else "No"
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("super_glue", "rte")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set


class ARC:
    def __init__(self, subset="easy", prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name

        if subset.lower() == "easy":
            self.subset = "ARC-Easy"
        elif subset.lower() == "challenge":
            self.subset = "ARC-Challenge"
        else:
            self.subset = subset  # fallback for robustness

        self.prompt_template = prompt_template or "Question: {question}\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("ai2_arc", self.subset)
        for split in ['train', 'validation', 'test']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            choices = {label: text for label, text in zip(d["choices"]["label"], d["choices"]["text"])}
            for opt in ["A", "B", "C", "D", "E"]:
                if opt not in choices:
                    choices[opt] = ""

            prompt = self.prompt_template.format(
                question=d["question"],
                A=choices.get("A", ""),
                B=choices.get("B", ""),
                C=choices.get("C", ""),
                D=choices.get("D", ""),
                E=choices.get("E", "")
            )

            answer = d["answerKey"]
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("ai2_arc", self.subset)
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set



