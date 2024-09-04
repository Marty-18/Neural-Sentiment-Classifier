import argparse
from trainer import Trainer

def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('GPT2 Model fine-tuned on Austen dataset')

    #Add arguments for data options
    parser.add_argument('--data', default='IMDB_Dataset.csv', help='path to data directory')
    parser.add_argument('--batch-size', default=8, type=int, help='maximum number of datapoints in a batch')
   # parser.add_argument('--max-input-length', default=100, type=int, help='maximum length of data input sequence' )
   #parser.add_argument('--train-size', default=0.9, type=float, help='percentage of data for training split')

    #Model arguments 
    parser.add_argument('--model', default='TextClassifier', help='model name')
    
    #Optimization arguments
   # parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
  # eval every 250 steps  
    parser.add_argument('--log-every', default=100, type=int, help='every number of steps after which training stats are shown.')
    parser.add_argument('--epochs', default=4, type=int, help='train until specified epoch')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
   # parser.add_argument('--eps', default=1e-8, type=float, help='Adam’s epsilon for numerical stability')

    #Saving and loading checkpoint arguments
   #parser.add_argument('--check-dir', default='Checkpoints', help='path to directory to save checkpoints')
   # parser.add_argument('--restore-file', type=str, help='name of the folder inside the checpoints folder which indicates the epoh checkpoint you want to load') 
   # parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
   # parser.add_argument('--load-checkpoint', default=False, type=bool, help='whether to load the model from checkpoint')


    #Save model arguments
    parser.add_argument('--output-dir', default='GPT2_fine_tuned_Austen', help='path to save logs')

    args = parser.parse_args()
    return args




def main():
    #load data
    #load model
    #optimizer
    #loss
    #scheduler
    trainer = Trainer(args) #in right order
    trainer.train()
    #test as well?





if __name__=='__main__':
    args=get_args()

    main(args)