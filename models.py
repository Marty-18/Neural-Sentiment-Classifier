import torch
from torch import nn
# define a fully connected model with an embedding layer for classification
class TextClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=False)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.act1 = nn.ReLU()#nn.Tanh()#nn.ReLU()
        #self.fc2 = nn.Linear(self.hidden_dim, hidden_dim)
        #self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()


  def forward(self, x):
      embedded = self.embedding(x)
      #print('embedded shape', embedded.shape) [batch, sequence_len, embedding_dim]
      avg_pool = torch.mean(embedded, dim=1)
      #print('avg_pool shape', avg_pool.shape) #[batch, embedding_dim]
      h1 = self.act1(self.fc1(avg_pool))
    ###  h2 = self.act2(self.fc2(h1))
     # print('h1 shape', h1.shape)
     # print('h2 shape', h2.shape)

      out_sig = self.sig(self.output(h1))
    # print('out_sig shape', out_sig.shape)

      # reshape to be batch_size first
      out_sig = out_sig.view(len(x), -1) #batch_size, -1 or len(x)?
      #print('out sig', out_sig.shape)

      # get last prediction for each batch
      out_sig = out_sig[:, -1]

      return out_sig



