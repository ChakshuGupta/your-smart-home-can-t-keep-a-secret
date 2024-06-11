import torch
import torch.nn as nn

class LstmModel(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, input_dim, hidden_dim, layer_dim, output_dim):
        super(LstmModel, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim

        self.embedding =  nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0.5)
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=2)
    

    def forward(self, features):
        
        embedded_dport = self.embedding(features[:,:,1].to(torch.long))
        # # Replace the original dport column in the features with the pca encoded one
        input = torch.cat([embedded_dport,features[:,:,1:]],dim=2)

        h0 = torch.zeros(self.output_dim, input.size(0), self.hidden_size)
        c0 = torch.zeros(self.output_dim, input.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(input, (h0, c0))
        lstm_out = self.fc(lstm_out[:, -1, :])
         # Softmax
        output = self.softmax(lstm_out)

        return output
