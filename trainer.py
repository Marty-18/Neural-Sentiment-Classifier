import  torch
class Trainer():
    def __init__(self, model, train_data, val_data, batch_size, learning_rate, num_epochs, optimizer, scheduler, loss, output_dir, log_steps):
       self.stats_dict = {'train_loss': [], 'train_acc': [], 'train_f1_score': [], 'train_time': [], 'val_loss': [], 'val_acc': [], 'val_f1_score': [], 'val_time': []}
       #self.model = model
       #self.train_data = train_data
       #self.val_data = val_data
       self.batch_size = batch_size
       self.lr = learning_rate
       self.num_epochs = num_epochs
       self.optimizer = optimizer #pick the right one so pass a string here?
       self.scheduler = scheduler #same as optimier?
       self.loss = loss #do i need to instantiate all of these? same as the two above?
       self.patience = 10 # number of consecutive epochs were val f1 score doesn't improve
       self.bad_epochs = 0 # counter to track epchs with no val f1 score improvement
       self.total_training_time = 0
       
    def train_epoch():
        pass

    def eval_epoch():
        pass

    def save_model():
        "not implemented"
        #save best model at end? save after n epochs?
        pass

    def train():
        # log training params
        print(f'Training model: {model}, with learning rate: {self.lr}, batch size: {self.batch_size}, hidden dim: {model.hidden_dim}, embedding dim: {model.embedding_dim} for {self.num_epochs} epochs.')


    