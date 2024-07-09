# List all the constant values here

# Number of processes
NUM_PROCS = 4

# Size of the sliding window
WIN_SIZE = 20

# Configuration for the LSTM model
class Config:
    hidden_dim = 64
    embedding_dim = 30
    num_layer = 2
    dropout_rate = 0.5
    learning_rate = 0.001
    num_epochs = 15
    input_dim = 40 # Fingerprint size (11) - subtract dport (1) + embedding dim (30)
    num_embeddings = 65535 # Max possible value for ports