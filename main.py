import argparse

def get_args():
    """
    Defines training-specific hyper-parameters.

    """

    parser = argparse.ArgumentParser('GPT2 Model fine-tuned on Austen dataset')

    #Add arguments for data options
    parser.add_argument('--data', default='austen_dataset.pkl', help='path to data directory')
    parser.add_argument('--batch-size', default=8, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-input-length', default=100, type=int, help='maximum length of data input sequence' )
    parser.add_argument('--train-size', default=0.9, type=float, help='percentage of data for training split')

    #Model generation arguments for generation during evaluation
    parser.add_argument('--max-output-length', default=200, help='maximum length of output sequence during evaluation' )
    parser.add_argument('--top-k', default=50, type=int, help='number of highest probability vocabulary tokens to keep for top-k-filtering')
    #only the set of tokens with proability that add up to top_p or higher are used for generation
    parser.add_argument('--top-p', default=0.95, type=float, help='probability threshold to select the set of tokens used for generation')

    #Model arguments 
    parser.add_argument('--model', default='gpt2', help='model name from HuggingFace')
    parser.add_argument('--qlora', default=False, type=bool, help='training using QLORA peft method instead of full fine-tuning')
    
    #QLORA config arguments
    parser.add_argument('--rank_lora', default=64, type=int, help='rank of lo rank matrices in LORA')
    parser.add_argument('--alpha_lora', default=16, type=int, help='alpha scaling parameter in LORA')
    parser.add_argument('--targets_lora', default =['c_attn'], type=str, nargs='+', help='list of modules to apply adapters to for LORA')

    #Optimization arguments
    parser.add_argument('--warmup-steps', default=1e2, type=float, help='number of warm up steps for learing rate scheduler')
    parser.add_argument('--sample-every', default=100, type=int, help='every number of steps after which a random sample is outputted')
    parser.add_argument('--epochs', default=4, type=int, help='train until specified epoch')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='Adam’s epsilon for numerical stability')

    #Saving and loading checkpoint arguments
    parser.add_argument('--check-dir', default='Checkpoints', help='path to directory to save checkpoints')
    parser.add_argument('--restore-file', type=str, help='name of the folder inside the checpoints folder which indicates the epoh checkpoint you want to load') 
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--load-checkpoint', default=False, type=bool, help='whether to load the model from checkpoint')


    #Save model arguments
    parser.add_argument('--output-dir', default='GPT2_fine_tuned_Austen', help='path to save logs')

    args = parser.parse_args()
    return args

class Trainer():
    def __init__(self, )
       self.stats_dict = {'train_loss': [], 'train_acc': [], 'train_f1_score': [], 'train_time': [], 'val_loss': [], 'val_acc': [], 'val_f1_score': [], 'val_time': []}
def main():
    pass




if __name__=='__main__':
    args=get_args()
   
    main(args)