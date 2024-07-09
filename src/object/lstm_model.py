import torch
import torch.nn as nn

class LstmModel(nn.Module):

    def __init__(self, config, output_dim, bidirectional=False, device="cpu"):
        super(LstmModel, self).__init__()

        self.config = config
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding =  nn.Embedding(self.config.num_embeddings,
                                       self.config.embedding_dim, device=self.device)
        
        self.lstm = nn.LSTM(self.config.input_dim, self.config.hidden_dim, 
                            self.config.num_layer, batch_first=True,
                            dropout=self.config.dropout_rate,
                            bidirectional=self.bidirectional,
                            device=self.device)
        
        # Fully connected layer
        if self.bidirectional:
            self.fc = nn.Linear(self.config.hidden_dim * 2,
                                self.output_dim, device=self.device, bias= True)
        else:
            self.fc = nn.Linear(self.config.hidden_dim,
                                self.output_dim, device=self.device, bias= True)
            

    def forward(self, features):                
        embedded_dport = self.embedding(features[:,:,1].to(torch.long))
        # # Replace the original dport column in the features with the embedded ones
        input = torch.cat([embedded_dport,features[:,:,1:]],dim=2)
        input = input.float() #  To fix the error that LSTM was expecting float32 but got float64

         # Fully-connected
        if self.bidirectional:
            h0 = torch.zeros(self.config.num_layer * 2, input.size(0),
                             self.config.hidden_dim, dtype=torch.float32,
                             device=self.device)
            c0 = torch.zeros(self.config.num_layer * 2, input.size(0),
                             self.config.hidden_dim, dtype=torch.float32,
                             device=self.device)
        else:
            h0 = torch.zeros(self.config.num_layer, input.size(0),
                             self.config.hidden_dim, dtype=torch.float32,
                             device=self.device)
            c0 = torch.zeros(self.config.num_layer, input.size(0),
                             self.config.hidden_dim, dtype=torch.float32,
                             device=self.device)
        
        lstm_out, _ = self.lstm(input, (h0, c0))

        lstm_out = self.fc(lstm_out[:, -1, :])

        return lstm_out
