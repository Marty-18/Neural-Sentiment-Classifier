import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split

def load_process_data(file):
    """
    Load the raw CSV file, preprocess the data and split into train, eval, test. 
    
    Param: file name.
    Return: three splits of the data.
    """
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        print(f'File named {file} not found. Please try a different file path or name.')

     # get rid of duplicates in the data 
    data = data.drop_duplicates()
       
    # convert labels to numerical features, 1 for the positive class, 0 for the negative one
    data.loc[data['sentiment'] == 'positive', 'sentiment'] = 1 #positive class
    data.loc[data['sentiment'] == 'negative', 'sentiment'] = 0 #negative class

    # split data into train, val, test
    train_val_data, test_data = train_test_split(data, test_size=0.10, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=len(test_data), random_state=42)
    print(f'Split the dataset into training, validation and testing split of length {len(train_data)}, {len(val_data)}, {len(test_data)}, respectively.')
    
    return train_data, val_data, test_data


def plot_metrics(stats_dict):
    epochs = range(len(stats_dict['train_loss']))
    measures = ['loss', 'accuracy', 'f1_score']

    for measure in measures:
        # blue dot line for training
        plt.plot(epochs, stats_dict['train_'+measure], 'bo', label='Training '+measure.title())
        # solid blue line for validation
        plt.plot(epochs, stats_dict['val_'+measure], 'b', label='Validation '+measure.title())
        plt.title('Training and Validation '+measure.title())
        plt.xlabel('Epochs')
        plt.ylabel(measure.title())
        plt.legend()
        plt.show()
