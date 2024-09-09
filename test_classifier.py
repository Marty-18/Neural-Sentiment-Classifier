import unittest
import numpy as np
import pandas as pd
from trainer import Trainer
from models import TextClassifier
from dataset import DataVocab, ClassifierTextDataset
from utils import load_process_data
from torchtext.data.utils import get_tokenizer

class ClassifierTest(unittest.TestCase):
    
    def test_load_process_data(self):
        data_train, data_val, data_test = load_process_data('./IMDB_Dataset.csv')
        self.assertIsInstance(data_train, pd.DataFrame)
        self.assertIsInstance(data_val, pd.DataFrame)
        self.assertIsInstance(data_val, pd.DataFrame)
        # check val and test data are equal length
        self.assertEqual(len(data_val), len(data_test))

        # check labels are encoded as integers 
        self.assertIsInstance(data_train.sentiment.iloc[4000], int)
        self.assertIsInstance(data_val.sentiment.iloc[100], int)
        self.assertIsInstance(data_test.sentiment.iloc[500], int)

        self.assertEqual(data_train.sentiment.iloc[4000], 0)
        self.assertEqual(data_val.sentiment.iloc[500], 1)
        self.assertEqual(data_test.sentiment.iloc[2000], 0)
       
        # check label distribution in each split
        self.assertAlmostEqual(len(data_train[data_train['sentiment'] == 'positive']), len(data_train[data_train['sentiment'] == 'negative']))

    def test_vocab(self):
        vocab_size = 10000
        train_data, _, _ = load_process_data('./IMDB_Dataset.csv')
        tokenizer = get_tokenizer("basic_english")
        data_vocab = DataVocab(train_data, tokenizer, vocab_size=vocab_size)
        count_dict = data_vocab.dict_count
        #print(count_dict)
        vocab = data_vocab.vocab
        
        # test dictionary count is of correct length
        self.assertEqual(len(count_dict), vocab_size)

        # check punctuation and stop words are not in dictionary
        with self.assertRaises(KeyError):
            count_dict['.']
        with self.assertRaises(KeyError):
            count_dict['the']
        with self.assertRaises(KeyError):
            count_dict['?']
        with self.assertRaises(KeyError):
            count_dict['movie']
        with self.assertRaises(KeyError):
            count_dict['this']


        # test vocabulary is of correct length
        self.assertEqual(len(vocab), vocab_size+2)

        # test vocabulary indices for special tokens are correct
        self.assertEqual(vocab['<pad>'], 0)
        self.assertEqual(vocab['<unk>'], 1)
        self.assertEqual(vocab.get_default_index(), vocab['<unk>'])

        # check encodings of a sentence against expected encoding
        sentence = ['hello', ',', 'random', 'sentence', 'sdfi', '.']
        indeces = []
        for word in sentence:
            indeces.append(vocab[word])

        #print(indeces)
        self.assertEqual(vocab(sentence), indeces)


                               

    def test_dataset(self):
        pass 
    
    def test_model(self):
        pass

    def trainer_test(self):
       # model = TextClassifier(vocab_size=10002, embedding_dim=16, hidden_dim=16, output_dim=1)
       # t = Trainer(model=model, train_data, val_data, batch_size, learning_rate, num_epochs, optimizer, scheduler, loss, output_dir, log_steps)
      #  t.train()
        pass

if __name__=='__main__':
    unittest.main()
  
   # train_data = 
   # trainer_test(model)