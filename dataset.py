import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from string import punctuation
from collections import OrderedDict

         
nltk.download('stopwords')      

class DataVocab:
    """
    Create vocabulary to encode the whole dataset based on the training data to avoid data leaks. 
    """
    def __init__(self, train_data, tokenizer, vocab_size=10000, max_len=267):
        self.tokenizer = tokenizer
        # create the vocabulary dictionary
        dict_count = self.count_vocab(train_data)
        # use the dictionary to make a mapping between integers and words, add special characters for padding and unknown words
        self.vocab = vocab(dict_count, specials=["<pad>", "<unk>"]) 
        # set the index of words not in the vocabulary to be equal to the <unk> index
        self.vocab.set_default_index(vocabulary["<unk>"])
        
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

            # return only the top 10 000 words
            top_words = OrderedDict([w for w in sorted_count_dict[:10000]])

            return top_words       


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
    
    def pad_collate():
       