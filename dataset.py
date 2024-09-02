import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nltk
         
nltk.download('stopwords')      

class DataVocab:
    """
    Create vocabulary to encode the whole dataset based on the training data to avoid data leaks. 
    """
    def __init__(self, train_data, tokenizer, vocab_size=10000, max_len=267):
         self.vocab = create_vocab(count_vocab(train_data['review']))
         
        
        def count_vocab(self):
            dict_count = {}
            for text in data_iter:
                text = tokenizer(text)

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
    
    def pad_collate():
       