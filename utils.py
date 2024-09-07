import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file):
    """
    Load the raw CSV file, preprocess the data and split into train, tval, test. 
    
    Param: file name.
    Return: three splits of the data.
    """
    try:
        data = pd.load_csv(file)
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