import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from transformers import BertModel
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

MAX_LEN = 64

# loading all data into memory
corpus_movie_conv = 'movie_conversations.txt'
corpus_movie_lines = 'movie_lines.txt'
with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()
with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

# splitting text using special lines
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

# generate question answer pairs
pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []

        if i == len(ids) - 1:
            break

        first = lines_dic[ids[i]].strip()
        second = lines_dic[ids[i + 1]].strip()

        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)


# Create 'data' folder for text files if it doesn't exist
os.makedirs('data', exist_ok=True)
text_data = []
file_count = 0

for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # Once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

# Save any remaining data to a new file
if text_data:
    with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))

# Find all files in the `data` directory for training
paths = [str(x) for x in Path('./data').glob('**/*.txt')]

# Create a new WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(lowercase=True))

# Set up whitespace pre-tokenizer to split text into words
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Define the special tokens
special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']

# Set up a WordPiece trainer
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    min_frequency=5,
    special_tokens=special_tokens,
    limit_alphabet=1000,  # Optional: Limit size of character alphabet
)

# Train the tokenizer on the text files
tokenizer.train(files=paths, trainer=trainer)

# Save the trained tokenizer models
os.makedirs('./bert-it-1', exist_ok=True)
tokenizer.model.save('./bert-it-1', 'bert-it')

# Load the tokenizer with Hugging Face's `BertTokenizer` for compatibility
bert_tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

