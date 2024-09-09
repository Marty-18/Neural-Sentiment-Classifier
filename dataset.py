import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from string import punctuation
from collections import OrderedDict

         
nltk.download('stopwords')      

class DataVocab:
    """
    Create vocabulary to encode the whole dataset based on only the training data to avoid data leaks from training to test. 
    """
    def __init__(self, train_data, tokenizer, vocab_size=10000):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # create the vocabulary dictionary
        self.dict_count = self.count_vocab(train_data)
        # use the dictionary to make a mapping between integers and words, add special characters for padding and unknown words
        self.vocab = vocab(self.dict_count, specials=["<pad>", "<unk>"]) 
        # set the index of words not in the vocabulary to be equal to the <unk> index
        self.vocab.set_default_index(self.vocab["<unk>"])
        
    def count_vocab(self, data):
        dict_count = {}
        for text in data:
            text = self.tokenizer(text)

            # remove stop words and punctuation
            stop_words = set(stopwords.words('english'))
            filtered_text = [w for w in text if not w in stop_words and not w in punctuation]
            #print(filtered_text)

            for word in filtered_text:
                try:
                    dict_count[word] += 1
                except:
                    dict_count[word] = 1


        # sort dictionary by value
        sorted_count_dict = sorted(dict_count.items(), key=lambda x:x[1], reverse=True)
        #print(type(sorted_count_dict))
        #print(sorted_count_dict[0][0])

        # return only the top n words
        top_words = OrderedDict([w for w in sorted_count_dict[:self.vocab_size]])

        return top_words       


class ClassifierTextDataset(Dataset):
    """ 
    subsetting the torch dataset class to make custom dataset
    """
    def __init__(self, data, tokenizer, vocab, max_len=267): # vocab_size=10000, max_len=267):
        self.data = data
        #self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]
    
    def pad_collate(self, batch):
        """"
        create custom pad collate function to format the data for batch training by padding and trimming every input in the batch to the same length
        """
        text_list = []
        label_list = []
        for text, label in batch:
            text_list.append(F.pad(text, pad=(0,self.max_len-len(text)), mode='constant', value=self.vocab.__getitem__('<pad>')))
            label_list.append(label)
            label_list = torch.tensor(label_list)
            text_pad = torch.stack(text_list)

        return [(text_pad.squeeze(), label_list)]

       