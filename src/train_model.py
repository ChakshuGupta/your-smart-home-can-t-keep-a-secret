import torch
from torch.optim.adam import Adam

from src.object.lstm_model import LstmModel
from src.util import make_dataset_iterable, convert_to_tensor

class Config:
    hidden_dim = 64
    embedding_dim = 30
    num_layer = 2
    dropout_rate = 0.5
    learning_rate = 0.0001
    num_epochs = 15
    input_dim = 34 # Fingerprint size (5) - subtract dport (1) + embedding dim (30)


def train_lstm_model(train_features, train_labels):

    train_x, train_y, label_mapping = convert_to_tensor(train_features, train_labels)
    
    num_embeddings = train_features["dport"].max()+1 # To make sure the max value is included in the embedding
    output_dim = len(label_mapping)
    
    print(train_x.shape, train_y.shape, num_embeddings)
    print(label_mapping)

    train_dataloader = make_dataset_iterable(train_x, train_y)
    
    config = Config()
    lstm_model = LstmModel(config, num_embeddings, output_dim)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index in the loss calculation
    optimizer = Adam(lstm_model.parameters(), lr=config.learning_rate)

    total_loss = 0
    # Training loop
    for epoch in range(config.num_epochs):
        lstm_model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_dataloader:
            print(x_batch[:,0].max())
            optimizer.zero_grad()
            outputs = lstm_model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_dataloader)
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss}')
    
    print("Train total loss: %5f" % (total_loss/config.num_epochs))

    # Save the model checkpoint
    torch.save(lstm_model.state_dict(), 'basic_lstm_model_checkpoint.pth')