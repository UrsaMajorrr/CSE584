import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LLM_datasets import LLMDataset
from transformers import BertTokenizer
import pandas as pd
from model_arch import LLMClassifier

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('datasets/llm_test.csv')

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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = LLMDataset(df["combined_text"].to_list(), df["model"].to_list(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Recreate model and load in weights
    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    hidden_dim = 256
    output_dim = 7  # Can change this if more LLMs are added to dataset
    num_layers = 2
    dropout = 0.3

    model = LLMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout)
    model.load_state_dict(torch.load('model_weights/model_12.pth', weights_only=True))
    model.to(device)

    model.eval()
    all_predictions = []
    all_labels = []

    # Now, loop through different test prompts
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            # Forward pass through the model
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1) # Size 1xbatch_size (16 in this case)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            print(f"Predicted: {predicted.cpu().numpy()}, Actual: {labels.cpu().numpy()}")

    correct = 0
    for i, j in zip(all_predictions, all_labels):
        if i == j:
            correct += 1
    accuracy = correct / len(all_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    
