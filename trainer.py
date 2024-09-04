import  torch
class Trainer():
    def __init__(self, model, train_data, val_data, batch_size, learning_rate, num_epochs, optimizer, scheduler, loss):
       self.stats_dict = {'train_loss': [], 'train_acc': [], 'train_f1_score': [], 'train_time': [], 'val_loss': [], 'val_acc': [], 'val_f1_score': [], 'val_time': []}
       #self.model = model
       #self.train_data = train_data
       #self.val_data = val_data
       self.batch_size = batch_size
       self.lr = learning_rate
       self.num_epochs = num_epochs
       self.optimizer = optimizer
       self.scheduler = scheduler
       self.loss = loss #do i need to instantiate all of these?

    def train_epoch():
        pass

    def eval_epoch():
        pass

    def train():
        # log training params
        print(f'Training model: {model}, with learning rate: {self.lr}, batch size: {self.batch_size}, hidden dim: {hidden_dim}, embedding dim: {embedding_dim} for {self.num_epochs} epochs.')


    