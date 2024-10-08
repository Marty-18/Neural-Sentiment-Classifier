import argparse
import torch
import numpy as np

from torch import nn
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, RandomSampler

from utils import load_process_data, plot_metrics
from dataset import DataVocab, ClassifierTextDataset
from trainer import Trainer
from models import TextClassifier

def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('Binary Neural Sentiment Classifier')

    # Add arguments for data 
    parser.add_argument('--data_file', default='IMDB_Dataset.csv', help='path to data directory')
    parser.add_argument('--batch-size', default=4, type=int, help='maximum number of datapoints in a batch')
    parser.add_argument('--vocab-size', default=10000, type=int, help='size of the model vocabulary')
    parser.add_argument('--max_input_length', default=267, type=int, help='maximum length of each review')

    # Model arguments 
    parser.add_argument('--model', default='TextClassifier', help='model name')
    parser.add_argument('--embedding_dim', default=16, type=int, help='dimensions for emebdding layer')
    parser.add_argument('--hidden_dim', default=16, type=int,  help='dimensions for hidden layers')
    
    # Optimization arguments
   # parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
    parser.add_argument('--log-every', default=500, type=int, help='every number of steps after which training stats are shown.')
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




def main(args):
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
                             batch_size=args.batch_size, collate_fn = train_dataset.pad_collate, drop_last=True)

    val_dataloader = DataLoader(val_dataset,
                              sampler=RandomSampler(val_dataset),
                             batch_size=args.batch_size, collate_fn = val_dataset.pad_collate, drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                              sampler=RandomSampler(test_dataset),
                             batch_size=args.batch_size, collate_fn = test_dataset.pad_collate, drop_last=True)
    # instantiate model and set manual seed
    torch.manual_seed(42)
    model = TextClassifier(len(vocab), args.embedding_dim, args.hidden_dim, output_dim)

    # define loss function, optimizer and lr scheduler
    criterion = nn.BCELoss() # binary cross entropy 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.epochs)
    
    trainer = Trainer(model, train_dataloader, val_dataloader, args.batch_size, args.lr, args.epochs, optimizer, scheduler, criterion=criterion, log_steps=args.log_every, output_dir=args.output_dir) 
    trainer.train()

    print('Best validation f1 score:', max(trainer.stats_dict['val_f1_score']))
    print(f"In Epoch: {np.argmax(trainer.stats_dict['val_f1_score'])+1}.")
    print("Best validation loss:", min(trainer.stats_dict['val_loss']))
    print(f"In Epoch: {np.argmin(trainer.stats_dict['val_loss'])+1}.")
    print('Best validation accuracy:', max(trainer.stats_dict['val_accuracy']))
    print(f"In Epoch: {np.argmax(trainer.stats_dict['val_accuracy'])+1}.")
    plot_metrics(trainer.stats_dict)
    
    # cross validation

    # evaluate on test set
    trainer.test(test_dataloader)



if __name__=='__main__':
    args=get_args()

    main(args)