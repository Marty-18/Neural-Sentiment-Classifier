import pandas as pd
import numpy as np
import spacy
from torch.utils.data import Dataset

class DataVocab:
    """
    Create vocabulary to encode the whole dataset based on the training data to avoid data leaks. 
    """
    def __init__(self, train_data, vocab_size=10000, max_len=267):
         self.vocab = create_vocab(count_vocab(train_data['review']))
        
        
        def count_vocab:
        
        def create_vocab:
        


class ClassifierTextDataset(Dataset):
    def __init__(self, data, tokenizer, vocab): # vocab_size=10000, max_len=267):
        self.data = data
        #self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]
    
    def pad_collare():
       