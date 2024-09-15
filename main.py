import argparse
import torch

from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, RandomSampler

from utils import load_process_data
from dataset import DataVocab, ClassifierTextDataset
from trainer import Trainer
from models import TextClassifier

def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('GPT2 Model fine-tuned on Austen dataset')

    # Add arguments for data 
    parser.add_argument('--data_file', default='IMDB_Dataset.csv', help='path to data directory')
    parser.add_argument('--batch-size', default=8, type=int, help='maximum number of datapoints in a batch')
    parser.add_argument('--vocab-size', default=10000, type=int, help='size of the model vocabulary')
    parser.add_argument('--max_input_length', default=267, type=int, help='maximum length of each review')


   # parser.add_argument('--max-input-length', default=100, type=int, help='maximum length of data input sequence' )
   #parser.add_argument('--train-size', default=0.9, type=float, help='percentage of data for training split')

    # Model arguments 
    parser.add_argument('--model', default='TextClassifier', help='model name')
    parser.add_argument('--embedding_dim', default=16, type=int, help='dimensions for emebdding layer')
    parser.add_argument('--hidden_dim', default=16, type=int,  help='dimensions for hidden layers')
    
    # Optimization arguments
   # parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
  # eval every 250 steps  
    parser.add_argument('--log-every', default=250, type=int, help='every number of steps after which training stats are shown.')
    parser.add_argument('--epochs', default=30, type=int, help='train until specified epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
   # parser.add_argument('--eps', default=1e-8, type=float, help='Adam’s epsilon for numerical stability')

    #Saving and loading checkpoint arguments
   #parser.add_argument('--check-dir', default='Checkpoints', help='path to directory to save checkpoints')
   # parser.add_argument('--restore-file', type=str, help='name of the folder inside the checpoints folder which indicates the epoh checkpoint you want to load') 
   # parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
   # parser.add_argument('--load-checkpoint', default=False, type=bool, help='whether to load the model from checkpoint')


    #Save model arguments
    parser.add_argument('--output-dir', default='Sentiment_classifier_model', help='path to save logs')

    args = parser.parse_args()
    return args




def main():
    output_dim = 1
    # load and preprocess data
    data_train, data_val, data_test = load_process_data(args.data_file)

    # load tokenizer
    tokenizer = get_tokenizer("basic_english")
    vocab = DataVocab(data_train, tokenizer, vocab_size=args.vocab_size).vocab

    # make dataset into custom dataset
    train_dataset = ClassifierTextDataset(data_train, tokenizer, vocab, args.max_input_length)
    val_dataset = ClassifierTextDataset(data_val, tokenizer, vocab, args.max_input_length) 
    test_dataset = ClassifierTextDataset(data_test, tokenizer, vocab, args.max_input_length)
    
    # preaprare data for training using torch dataloaders with batching and padding
    train_dataloader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                             batch_size=args.batch_size, collate_fn = train_dataset.pad_collate(), drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                              sampler=RandomSampler(val_dataset),
                             batch_size=args.batch_size, collate_fn = val_dataset.pad_collate(), drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                              sampler=RandomSampler(test_dataset),
                             batch_size=args.batch_size, collate_fn = test_dataset.pad_collate(), drop_last=True)
    # instantiate model
    model = TextClassifier(args.vocab_size, args.embedding_dim, args.hidden_dim, output_dim)

    # define loss function, optimizer and lr scheduler
    criterion = nn.BCELoss() # binary cross entropy 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.epochs)
    
    trainer = Trainer(model, train_dataloader, val_dataloader, args.batch_size, args.lr, args.epochs, optimizer, scheduler, args.loss, args.output_dir, args.log_every) 
    trainer.train()
    #test as well?





if __name__=='__main__':
    args=get_args()

    main(args)