"""
Script that holds the classes for creating datasets and dataloaders

Author: Kade Carlson
Date: 09/30/24
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

class LLMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        texts = self.texts[index]
        labels = self.labels[index]
        
        tks = self.tokenizer(
            texts,
            max_length=100,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tks['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'label': torch.tensor(labels, dtype=torch.long)
        }