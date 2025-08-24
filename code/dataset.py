import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import math

# 蛋白质序列索引映射字典
protein_residue2idx = {
    '[PAD]': 0,
    'X': 0,
    '[CLS]': 1,
    '[SEP]': 2,
    '[MASK]': 3,  # 添加MASK标记
    'A': 4,   # Alanine
    'C': 5,   # Cysteine
    'D': 6,   # Aspartic acid
    'E': 7,   # Glutamic acid
    'F': 8,   # Phenylalanine
    'G': 9,   # Glycine
    'H': 10,  # Histidine
    'I': 11,  # Isoleucine
    'K': 12,  # Lysine
    'L': 13,  # Leucine
    'M': 14,  # Methionine
    'N': 15,  # Asparagine
    'P': 16,  # Proline
    'Q': 17,  # Glutamine
    'R': 18,  # Arginine
    'S': 19,  # Serine
    'T': 20,  # Threonine
    'V': 21,  # Valine
    'W': 22,  # Tryptophan
    'Y': 23   # Tyrosine
}

# 数据预处理函数
def transform_protein_to_index(sequences, residue2idx):
    token_index = []
    for seq in sequences:
        seq_id = [residue2idx.get(residue, 0) for residue in seq] 
        token_index.append(seq_id)
    return token_index

def pad_sequence(token_list, max_len=16):
    data = []
    for i in range(len(token_list)):
        token_list[i] = [protein_residue2idx['[CLS]']] + token_list[i] 
        if len(token_list[i]) > max_len:  
            token_list[i] = token_list[i][:max_len]
        else:  
            n_pad = max_len - len(token_list[i])
            token_list[i].extend([protein_residue2idx['[PAD]']] * n_pad)
        data.append(token_list[i])
    return data

def read_protein_sequences_from_fasta(file_path):
    sequences = []
    labels = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                if 'pos' in line:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences, labels

def load_data_from_txt(file_path, max_len=16):
    sequences, labels = read_protein_sequences_from_fasta(file_path)
    indexed_sequences = transform_protein_to_index(sequences, protein_residue2idx)
    padded_sequences = pad_sequence(indexed_sequences, max_len=max_len)
    return padded_sequences, labels

def load_features_from_txt(feature_file_path):
    features = np.loadtxt(feature_file_path)
    return features

# 数据集定义
class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, features, labels):
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )




