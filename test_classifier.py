import unittest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from trainer import Trainer
from models import TextClassifier
from dataset import DataVocab, ClassifierTextDataset
from utils import load_process_data
from torchtext.data.utils import get_tokenizer

class ClassifierTest(unittest.TestCase):
    
    @unittest.skip("already tested")
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
    
    @unittest.skip("already tested")
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
        indeces= []
        for word in sentence:
            indeces.append(vocab[word])

       
        self.assertEqual(vocab(sentence), indeces)
    
    @unittest.skip("already tested")          
    def test_classifier_text_dataset(self):
        vocab_size = 10000
        max_len=267
        data_train, data_val, data_test = load_process_data('./IMDB_Dataset.csv')
        tokenizer = get_tokenizer("basic_english")
        vocab = DataVocab(data_train, tokenizer, vocab_size=vocab_size).vocab
        train_dataset = ClassifierTextDataset(data_train, tokenizer, vocab)
        val_dataset = ClassifierTextDataset(data_val, tokenizer, vocab) 
        test_dataset = ClassifierTextDataset(data_test, tokenizer, vocab)

        # chech length of datasets is the same as the initial splits
        self.assertEqual(len(train_dataset), len(data_train))
        self.assertEqual(len(val_dataset), len(data_val))
        self.assertEqual(len(test_dataset), len(data_test))

        # check number of dimensions and type of encodings and labels
        for text, label in train_dataset:
            #print(text)
            self.assertIsInstance(text, torch.LongTensor)
            self.assertIsInstance(label, torch.LongTensor)
            self.assertEqual(text.dim(), label.dim())

        # check number of dimensions and type of encodings and labels
        for text, labels in val_dataset:
            #print(text)
            self.assertIsInstance(text, torch.LongTensor)
            self.assertIsInstance(labels, torch.LongTensor)
            self.assertEqual(text.dim(), label.dim())
       
        # test collate function
        batch_size = 4
        # convert data to pytroch dataloaders with batching and padding 
        train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                             batch_size=batch_size, collate_fn = train_dataset.pad_collate, drop_last=True)

        val_dataloader = DataLoader(val_dataset,
                              sampler=RandomSampler(val_dataset),
                             batch_size=batch_size, collate_fn = val_dataset.pad_collate, drop_last=True)

        # check each batch is of the expected length and each input is long the maximum length
        for batch in train_dataloader:
           inputs = batch[0][0]
           labels = batch[0][1]
           self.assertEqual(len(inputs), batch_size)
           self.assertEqual(len(labels), batch_size)

           for sentence in inputs:
                self.assertEqual(len(sentence), max_len)
        
        for batch in val_dataloader:
           inputs = batch[0][0]
          # print(type(inputs))
          # print(inputs.shape)
           labels = batch[0][1]
          # print('labels', type(labels))
          # print(labels.shape)
           self.assertEqual(len(inputs), batch_size)
           self.assertEqual(len(labels), batch_size)

           for sentence in inputs:
                self.assertEqual(len(sentence), max_len)
    
   # @unittest.skip("already tested")
    def test_fc_model(self):
        vocab_size = 10002
        embedding_dim = 16
        hidden_dim = 16
        output_dim = 1
        # expected mdoel architecture
        model_architecture = ['embedding', 'fc1', 'act1', 'output_fc2', 'sig']
        dim_per_layer = [vocab_size*embedding_dim, embedding_dim*hidden_dim, hidden_dim, hidden_dim*output_dim, output_dim]
        # instantiate model
        model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
        model_params = [name for name, _ in model.named_children()]

        # check model architecture is as expected
        self.assertListEqual(model_params, model_architecture)
                
        # check dimensions per model parameter
        for param, layer_dim in zip(model.parameters(), dim_per_layer):
            self.assertEqual(param.nelement(), layer_dim)
        
        # test model prediction output is of correct shape
        example_input = torch.ones([4, 267]).to(torch.int64)
        expected_output = torch.ones([4])
        outputs = model(example_input)
        self.assertEqual(outputs.shape, expected_output.shape)
        
        


    def trainer_test(self):
       # model = TextClassifier(vocab_size=10002, embedding_dim=16, hidden_dim=16, output_dim=1)
       # t = Trainer(model=model, train_data, val_data, batch_size, learning_rate, num_epochs, optimizer, scheduler, loss, output_dir, log_steps)
      #  t.train()
        pass

if __name__=='__main__':
    unittest.main()
  
   # train_data = 
   # trainer_test(model)