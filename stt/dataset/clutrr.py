import json
import os
import random

import pandas as pd
from datasets import Dataset


class CLUTRR:
    def __init__(self, split_dir, prompt_template=None, few_shot_template=None, chat_template=False, model_name=None,
                 seed=42):
        self.split_dir = split_dir
        self.datasets = {'train': {}, 'val': {}, 'test': {}}
        self.seed = seed
        if prompt_template is None:
            ### Default prompt template
            self.prompt_template = "Read the following story about a family. \"{}\" Assume the relations described in the story are all true and based on your commonsense knowledge about family relationship, how is {} related to {}? Answer: {} is {} 's"
        else:
            self.prompt_template = prompt_template
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template}[/INST]"
        self.few_shot_template = few_shot_template
        self.model_name = model_name

    def load_data(self, train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.split_dir, f'{split}.json')
            data_json = json.load(open(data_dir, 'r'))
            if train_size != 0 and split == 'train':
                data_json = random.sample(data_json, train_size)
            if split == 'test':
                formatted_data = self.format_clutrr_prompt(data_json, append_label=False)
            else:
                formatted_data = self.format_clutrr_prompt(data_json, append_label=True)
            self.datasets[split] = formatted_data
        return self.datasets

    def format_clutrr_prompt(self, data_json, append_label):
        data = []
        for i in range(len(data_json)):
            story = data_json[i]['clean_story']
            ## Remove the irrelevant brackets from the story in the raw dataset
            story = story.replace('[', '').replace(']', '')
            persons = data_json[i]['query'][1:-1].replace('\'', '').split(',')
            per1 = persons[0]
            per2 = persons[1]
            gold_rel = data_json[i]['target_text']
            prompt_body = self.prompt_template.format(story, per2, per1, per2, per1)
            if append_label:
                if 'gemma' in self.model_name:
                    text = f"{prompt_body}\n{gold_rel}"
                else:
                    text = f"{prompt_body}\n{gold_rel}</s>"
                data.append({'text': text})
            else:
                text = f"{prompt_body}\n"
                data.append({'prompt': text, 'target_text': gold_rel})

        df = pd.DataFrame(data=data)
        df = df.astype('str')
        data = Dataset.from_pandas(df)
        return data

    def get_active_set(self, ratio=0.05):
        """
        Sample a fraction (default 3%) of the raw training data and preprocess it using test-like formatting,
        i.e. exactly as in the test set (append_label=False).
        """
        data_dir = os.path.join(self.split_dir, 'train.json')
        data_json = json.load(open(data_dir, 'r'))
        total_samples = len(data_json)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)
        print(f"Total samples: {total_samples}, Sample size: {sample_size}")
        sampled_json = random.sample(data_json, sample_size)
        active_set = self.format_clutrr_prompt(sampled_json, append_label=True)
        return active_set

    def get_dev_set(self, ratio=0.1, return_rest=True):
        """
        Split a portion of training data as dev set (formatted like test set), and return remaining as train.
        """
        data_dir = os.path.join(self.split_dir, 'train.json')
        data_json = json.load(open(data_dir, 'r'))
        total_samples = len(data_json)
        sample_size = max(1, int(total_samples * ratio))

        random.seed(self.seed)
        sampled_json = random.sample(data_json, sample_size)
        
        # remaining
        sampled_ids = set(id(ex) for ex in sampled_json)
        remaining_json = [ex for ex in data_json if id(ex) not in sampled_ids]

        # Dev set: format like test set (append_label=False) + required `text` field
        dev_data = self.format_clutrr_prompt(sampled_json, append_label=False)
        for i in range(len(dev_data)):
            dev_data[i]["text"] = dev_data[i]["prompt"] + dev_data[i]["target_text"]

        # Debug: print a few dev samples
        print(f"\n[Dev Set Sampling] total: {total_samples}, dev: {sample_size}, train_remain: {len(remaining_json)}")
        for i in range(min(3, len(sampled_json))):
            print(f"\n[Sample {i}]")
            print("Prompt:", dev_data[i]['prompt'])
            print("Label:", sampled_json[i]['target_text'])

        if return_rest:
            # Format remaining train set (with label)
            rest_train_data = self.format_clutrr_prompt(remaining_json, append_label=True)
            return dev_data, rest_train_data
        else:
            return dev_data
