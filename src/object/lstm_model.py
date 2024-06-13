import torch
import torch.nn as nn

class Config:
    hidden_dim = 64
    embedding_dim = 30
    num_layer = 2
    dropout_rate = 0.5
    learning_rate = 0.0001
    num_epochs = 15
    input_dim = 34 # Fingerprint size (5) - subtract dport (1) + embedding dim (30)
    num_embeddings = 65535 # Max possible value for ports


class LstmModel(nn.Module):

    def __init__(self, config, output_dim, bidirectional=False):
        super(LstmModel, self).__init__()

        self.config = config
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        self.embedding =  nn.Embedding(self.config.num_embeddings, self.config.embedding_dim)
        
        self.lstm = nn.LSTM(self.config.input_dim, self.config.hidden_dim, 
                            self.config.num_layer, batch_first=True,
                            dropout=self.config.dropout_rate,
                            bidirectional=self.bidirectional)
        
        # Fully connected layer
        if self.bidirectional:
            self.fc = nn.Linear(self.config.hidden_dim * 2, self.output_dim)
        else:
            self.fc = nn.Linear(self.config.hidden_dim, self.output_dim)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, features):
        
        embedded_dport = self.embedding(features[:,0].to(torch.long))
        # # Replace the original dport column in the features with the embedded ones
        input = torch.cat([embedded_dport,features[:,1:]],dim=1)
        input = input.float() #  To fix the error that LSTM was expecting float32 but got float64

        h0 = torch.zeros(self.config.num_layer, self.config.hidden_dim, dtype=torch.float32)
        c0 = torch.zeros(self.config.num_layer,  self.config.hidden_dim, dtype=torch.float32)
        lstm_out, (hidden, state) = self.lstm(input, (h0, c0))

         # Fully-connected
        if self.bidirectional:
            lstm_out = lstm_out.contiguous().view(-1, self.config.hidden_dim*2)  # Flatten the LSTM output
        else:
            lstm_out = lstm_out.contiguous().view(-1, self.config.hidden_dim)  # Flatten the LSTM output
        lstm_out = self.fc(lstm_out)

         # Softmax 
        output = self.softmax(lstm_out)
        return output
