"""
Script for training the model on the provided dataset

Author: Kade Carlson
Date: 09/30/2024
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LLM_datasets import LLMDataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from model_arch import LLMClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load in dataset
df = pd.read_csv('datasets/data.csv') # Must be changed if using different dataset

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

# Creating vvalidation data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=10)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = LLMDataset(train_df["combined_text"].to_list(), train_df["model"].to_list(), tokenizer)
val_dataset = LLMDataset(val_df["combined_text"].to_list(), val_df["model"].to_list(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Instantiate the model using hyperparameters
vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 256
output_dim = 7  # Can change this if more LLMs are added to dataset
num_layers = 2
dropout = 0.5
model = LLMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout)
model.to(device)

# Initialize loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Need some weight decay to prevent overfitting

num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Batch: {i+1} / {len(train_loader)}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids)
            loss = loss_func(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (j+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss.item():.4f}, Batch: {j+1} / {len(val_loader)}")
        
        val_accuracy = correct / total
        print(f"Val Accuracy: {val_accuracy:.4f}")

torch.save(model.state_dict(), 'model_weights/model_1.pth')
print("Saved model weights")

print("Done training")
