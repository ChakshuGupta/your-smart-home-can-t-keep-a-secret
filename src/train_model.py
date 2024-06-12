import os
import torch

from sklearn.metrics import classification_report
from torch.optim.adam import Adam

from src.object.lstm_model import LstmModel, Config
from src.util import make_dataset_iterable, convert_to_tensor


def train_lstm_model(train_features, train_labels, label_mapping, bidirectional=False):
    """
    Train the LSTM model
    """
    # Convert the dataframes to tensors
    x_train, y_train = convert_to_tensor(train_features, train_labels)

    # Final output dimension == number of devices    
    output_dim = len(label_mapping)
    
    # Load the data into a DataLoader and make it iterable by splitting
    # the data into batches
    train_dataloader = make_dataset_iterable(x_train, y_train)
    
    # Get the config for the LSTM model
    config = Config()
    
    # If the model file exists, load it and return the file
    if os.path.exists('lstm_model.sav'):
        lstm_model = LstmModel(config, output_dim, bidirectional)
        lstm_model.load_state_dict(torch.load('lstm_model.sav'))
        return lstm_model

    # Get the LSTM mddel class object
    lstm_model = LstmModel(config, output_dim, bidirectional)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index in the loss calculation
    optimizer = Adam(lstm_model.parameters(), lr=config.learning_rate)

    total_loss = 0
    # Training loop
    for epoch in range(config.num_epochs):
        # Set the model in training mode
        lstm_model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            # Forward propogation
            y_pred = lstm_model(x_batch)
            # Compute loss between predicted values and true values
            loss = criterion(y_pred, y_batch)
            total_loss += loss
            # Backpropagate loss
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_dataloader)
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss}')
    
    print("Train total loss: %5f" % (total_loss/config.num_epochs))

    # Save the model checkpoint
    torch.save(lstm_model.state_dict(), 'lstm_model.sav')
    # Return the trained model
    return lstm_model



def test_lstm_model(model, test_features, test_labels):
    """
    Test the LSTM model
    """
    model.eval()
    x_test, y_test = convert_to_tensor(test_features, test_labels)
    test_dataloader = make_dataset_iterable(x_test, y_test)
     
    correct = 0
    total = 0
    y_pred_total = []
    # Iterate through test dataset
    for x_batch, y_batch in test_dataloader:
        # Forward pass only to get logits/output
        outputs = model(x_batch)

        # Get predictions from the maximum value
        _, y_pred = torch.max(outputs.data, 1)
        
        y_pred_total.extend(y_pred)

        # Total number of labels
        total += y_batch.size(0)

        # Total correct predictions
        correct += (y_pred == y_batch).sum()

    accuracy = 100 * correct / total

    print(classification_report(
        y_true = y_test.numpy(),
        y_pred = y_pred_total,
        digits = 4,
        zero_division = 0,
    ))