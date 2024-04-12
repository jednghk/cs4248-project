from nltk.tokenize import word_tokenize
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import nltk
nltk.download('punkt')

column_labels = ['label', 'text']

# Load datasets
train_df = pd.read_csv('cooperator/data/fulltrain.csv', header=None, names=column_labels)
test_df = pd.read_csv('cooperator/data/balancedtest.csv', header=None, names=column_labels)

# Assuming the structure is ['label', 'text']
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# NLTK Tokenization
tokenized_texts = [word_tokenize(text) for text in train_texts]

# Building a vocabulary
def build_vocab(tokenized_texts):
    counter = Counter(token for text in tokenized_texts for token in text)
    return {word: i+2 for i, (word, _) in enumerate(counter.most_common())}, max(counter.values()) + 2

vocab, vocab_size = build_vocab(tokenized_texts)
print(f"Vocabulary Size: {vocab_size}")

# Texts to sequences
sequences = [[vocab[token] for token in text] for text in tokenized_texts]

# Padding sequences
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)

# Convert labels to a tensor
labels_tensor = torch.tensor(train_labels, dtype=torch.long)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_tensor, test_size=0.2, random_state=42)

# Creating data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
