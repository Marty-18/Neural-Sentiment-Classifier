# Neural-Sentiment-Classifier

Binary classifier for sentiment classification of the IMDB movie review dataset. The data contains 50K reviews of movies from the IMDB website. The model used for classification is a multilayer perceptron made up of an embedding layer as the input layer, the embeddings are then averaged using the mean across the sentence length dimension returning a tensor of size batch size and embedding dimension, giving an embedding tensor for each sentence in the batch. The embedding layer is followed  by a fully connected layer with a ReLU activation function and the final output layer is another fully connected layer but with an output sigmoid function of size 1 to return the label prediction. All other layers are of dimension 16, the model in total has 160321 trainable parameters. 

### Data pre-processing

The data was filtered removing duplicate reviews to reduce biased caused by seeing the same datapoint several times. This left 49582, 24698	negative reviews and 24884 positive reviews. There is a slight class imbalance so we used both accuracy and F1 score as evaluation metrics during training. The data was split into train, validation and test data containing 39664, 4959 and 4959 reviews respectively. The training data was used to create an encodings vocabulary containing only the 10 0000 most common tokens in the data after having removed stop words and words very common in this dataset like the word "movie". The tokens for unknown words and padding were added to this vocabulary. All the data was then tokenized and encoded using the vocabulary mapping created earlier. A collate function to format the data for batch training was created in which the data is padded or trimmed to the average data length of 267 tokens. 

### Performance

Here are the metrics and learning curves for the train and validation splits. 





