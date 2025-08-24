from dataset import *
from model import PepNet
import torch
import torch.utils.data as Data
import numpy as np
from dataset import load_data_from_txt, load_features_from_txt, MyDataSet
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

train_sequences, train_labels = load_data_from_txt('train.txt')
test_sequences, test_labels = load_data_from_txt('test.txt')


sequence_feature_train = load_features_from_txt('train_feature.txt')
sequence_feature_test = load_features_from_txt('test_feature.txt')

dataset_train = MyDataSet(train_sequences, sequence_feature_train, train_labels)
dataset_test = MyDataSet(test_sequences, sequence_feature_test, test_labels)


device = "cpu"
  

vocab_size = len(protein_residue2idx)
d_model = 256
d_ff = 1024
n_layers = 2
n_heads = 4
batch_size = 128
seq_feature_dim = sequence_feature_train.shape[1]
max_len = 16



train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


model = PepNet(vocab_size, d_model, seq_feature_dim, n_heads, d_ff, n_transformer_layers=n_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, sequence_features, labels = batch
        input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, sequence_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def calculate_metrics(outputs, labels):
    pred = outputs.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    acc = accuracy_score(labels_np, pred) 
    return {'Accuracy': acc}


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, sequence_features, labels = batch
            input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)

            outputs = model(input_ids, sequence_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['Loss'] = total_loss / len(data_loader)
    return metrics

n_epochs = 30
best_accuracy = 0.0


for epoch in range(n_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    if test_metrics["Accuracy"] > best_accuracy:
        best_accuracy = test_metrics["Accuracy"]
        print(f'Best model save as Epoch {epoch+1} with Accuracy: {best_accuracy:.4f}')


print(f'Completed!')