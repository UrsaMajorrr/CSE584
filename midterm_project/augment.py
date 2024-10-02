"""
Script for augmenting the dataset to add more data without querying the LLMs. May help with generalization

Author: Kade Carlson
Date: 10/02/24
"""

import pandas as pd
from textattack.augmentation import WordNetAugmenter

augmenter = WordNetAugmenter()
df = pd.read_csv("datasets/data.csv")
model_mapping = {
    "gpt-4o-mini": 0,
    "claude-2.1": 1,
    "command-r-plus": 2,
    "jamba-1.5-large": 3,
    "llama-3.2-1B": 4,
    "mistral-large-latest": 5,
    "gpt-neo-1.3B": 6
}

# Use a map to convert model labels to integers, model labels can continue to be added if new data is added
df["model"] = df["model"].map(model_mapping)
df["combined_text"] = df["input_prompt"] + df["completion"]

texts = df["combined_text"].to_list()
labels = df["model"].to_list()

augmented_text = []
augmented_labels = []

for k, (i, j) in enumerate(zip(texts, labels)):
    aug_vers = augmenter.augment(i)
    augmented_text.extend(aug_vers)
    augmented_labels.extend([j] * len(aug_vers))
    print(k)

augmented_frame = pd.DataFrame({"combined_text": augmented_text, "model": augmented_labels})
combine_frame = pd.concat([df, augmented_frame], ignore_index=True)
combine_frame.to_csv("datasets/augmented_zeroshot_data.csv")