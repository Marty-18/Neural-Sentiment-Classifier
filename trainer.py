import  torch
import time
from tqdm import tqdm
from python_utils import format_time
#from sklearn.metrics import f1_score
from torcheval.metrics.functional import binary_f1_score

class Trainer():
    def __init__(self, model, train_data, val_data, batch_size, learning_rate, num_epochs, optimizer, scheduler, criterion, log_steps, output_dir=None):
       self.stats_dict = {'train_loss': [], 'train_acc': [], 'train_f1_score': [], 'train_time': [], 'val_loss': [], 'val_acc': [], 'val_f1_score': [], 'val_time': []}
       self.model = model
       self.log_steps = log_steps
       self.train_data = train_data
       self.val_data = val_data
       self.batch_size = batch_size
       self.lr = learning_rate
       self.num_epochs = num_epochs
       self.optimizer = optimizer #pick the right one so pass a string here?
       self.scheduler = scheduler 
       self.criterion = criterion 
       self.patience = 10 # number of consecutive epochs were val f1 score doesn't improve
       self.bad_epochs = 0 # counter to track epchs with no val f1 score improvement
       self.total_training_time = 0

    def train_epoch(self, epoch):
        self.model.train()

        t0 = time.time()
        total_train_loss = 0
        total_train_samples = 0
        all_labels = []
        all_preds = []
        
        # display progress
        progress_bar = tqdm(self.train_data, desc=f'| Epoch {epoch}', leave=False, disable=False)

        # iterate over the training set
        for step, batch in enumerate(progress_bar):
            inputs = batch[0][0]
            labels = batch[0][1]

            # zero out the gradients from the previous batch
            self.optimizer.zero_grad()

            # model prediction
            outputs = self.model(inputs)

            # get model prediction
            pred = torch.round(outputs)

            assert outputs.shape == labels.shape

            # calculate loss for this batch
            loss = self.criterion(outputs, labels.float())
            batch_loss = loss.item()
   
            # backprogpagate loss
            loss.backward()
            self.optimizer.step()

            # accumulating training loss for every batch to get epoch training loss
            total_train_loss += loss.item() * len(inputs) 
            total_train_samples += len(inputs) #number of inputs per batch

            # accumulating all labels and predictions for each batch
            all_labels.append(labels)
            all_preds.append(pred.int())

            # print batch step, loss until now on training data and time elapsed every  n log_steps
            if step != 0 and step % self.log_steps == 0:
                elapsed = format_time(time.time()-t0)
                print(f' Step {step} of {len(self.train_data)}. Loss: {total_train_loss/total_train_samples:.4f}. Time: {elapsed}.')
        
       
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        assert len(all_labels) == len(all_preds)
        assert type(all_labels) == type(all_preds)
        
        # accuracy
        train_acc = (torch.sum(all_labels == all_preds))/total_train_samples
        print(f'Train accuracy for this epoch: {train_acc:.4f}')

        # f1 score
        #train_f1_score = f1_score(all_labels, all_preds, average='weighted') ground_truth, preds
        train_f1_score = binary_f1_score(all_preds, all_labels) #preds, ground_truth
        assert train_acc != train_f1_score
        
        # total time it took to train in this epoch
        final_elapsed = format_time(time.time()-t0)

        # average training loss in this epoch
        average_train_loss =  total_train_loss/total_train_samples

        # update the stats dictionary
        self.stats_dict['train_acc'].append(train_acc)
        self.stats_dict['train_f1_score'].append(train_f1_score)
        self.stats_dict['train_time'].append(final_elapsed)
        self.stats_dict['train_loss'].append(average_train_loss)

        return average_train_loss , train_f1_score, final_elapsed

    def eval_epoch(self):
        pass

    def save_model(self):
        "not implemented"
        #save best model at end? save after n epochs?
        pass

    def train(self):
        best_f1_val = 0
        # log training params
        print(f'Training model: {self.model}, with learning rate: {self.lr}, batch size: {self.batch_size}, hidden dim: {self.model.hidden_dim}, embedding dim: {self.model.embedding_dim} for {self.num_epochs} epochs.')

        # main training loop
        for epoch in range(self.num_epochs):
            epoch_t0 = time.time()
            epoch_train_loss, train_f1_score, train_time = self.train_epoch(epoch+1)
           #### avg_val_loss, f1_val_score, val_time = evaluate(val_dataloader, epoch+1)
            self.scheduler.step()
            epoch_time = time.time()-epoch_t0
            print(f"End of epoch {epoch+1}/{self.num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train F1 Score: {train_f1_score}, Learning rate scheduler: {self.optimizer.param_groups[0]['lr']}.")
            print(f"Epoch time: {format_time(epoch_time)}, Train time: {train_time}.")
            #print(f"End of epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train F1 Score: {train_f1_score}, Val F1 Score: {f1_val_score:.4f}, Learning rate scheduler: {optimizer.param_groups[0]['lr']}.")
            #print(f"Epoch time: {format_time(epoch_time)}, Train time: {train_time}, Validation time: {val_time}.")



    