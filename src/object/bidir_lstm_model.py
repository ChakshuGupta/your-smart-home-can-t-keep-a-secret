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

        self.embedding =  nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=0.5, bidirectional=True)

        self.label = nn.Linear(self.hidden_dim*2, self.output_dim)
    

    def forward(self):
        pass
