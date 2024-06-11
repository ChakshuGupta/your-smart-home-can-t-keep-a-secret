import torch
import torch.nn as nn

class BidirLstmModel(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, input_dim, hidden_dim, layer_dim, output_dim):
        super(BidirLstmModel, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0.5, bidirectional=True)

         # Fully connected layer
        self.fc = nn.Linear(self.config.hidden_dim * 2, self.output_dim)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, features):
        embedded_dport = self.embedding(features[:,0].to(torch.long))
        # # Replace the original dport column in the features with the pca encoded one
        input = torch.cat([embedded_dport,features[:,1:]],dim=1)
        input = input.float()

        h0 = torch.zeros(self.config.num_layer, self.config.hidden_dim, dtype=torch.float32)
        c0 = torch.zeros(self.config.num_layer,  self.config.hidden_dim, dtype=torch.float32)
        lstm_out, _ = self.lstm(input,(h0, c0))
         # Fully-connected
        lstm_out = lstm_out.contiguous().view(-1, self.config.hidden_dim*2)  # Flatten the LSTM output
        lstm_out = self.fc(lstm_out)
         # Softmax
        output = self.softmax(lstm_out)
        return output
