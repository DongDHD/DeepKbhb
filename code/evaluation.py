import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
from dataset import *
from model import PepNet  


device = "cpu"


test_sequences, test_labels = load_data_from_txt('test.txt')
sequence_feature_test = load_features_from_txt('test_feature.txt')


vocab_size = len(protein_residue2idx)
d_model = 256
d_ff = 1024
n_layers = 2
n_heads = 4
batch_size = 128
seq_feature_dim = sequence_feature_test.shape[1]
max_len = 16


dataset_test = MyDataSet(test_sequences, sequence_feature_test, test_labels)
test_loader = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

model = PepNet(vocab_size, d_model, seq_feature_dim, n_heads, d_ff, n_transformer_layers=n_layers).to(device)


best_model_path = "best_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, sequence_features, labels = batch
        input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)

        outputs = model(input_ids, sequence_features)
        probabilities = torch.softmax(outputs, dim=1)[:, 1] 
        predictions = outputs.argmax(dim=1) 

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())


accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred) 

mcc = matthews_corrcoef(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob) 

f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"测试集评估结果:")
print(f"ACC (准确率)   : {accuracy:.4f}")
print(f"SEN (灵敏度)   : {sensitivity:.4f}")

print(f"MCC (Matthews相关系数) : {mcc:.4f}")
print(f"AUC (曲线下面积) : {auc:.4f}")
print(f"F1 (F1分数) : {f1:.4f}")
print(f"Precision (精确率) : {precision:.4f}")
print(f"Recall (召回率) : {recall:.4f}")
print(f"Accuracy (准确率) : {acc:.4f}")

