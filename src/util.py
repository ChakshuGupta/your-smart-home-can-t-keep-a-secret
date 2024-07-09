import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.constants import NUM_PROCS


def load_device_file(device_file):
   """
   Load the mapping between devices and mac addresses
   """
   file_data = open(device_file, "r")
   device_mac_map = {}
   
   for line in file_data:
       if line.strip() == "":
           continue
       device = line.split(",")[0]
       mac = line.split(",")[1]
       device_mac_map[mac.strip()] = device.strip()

   return device_mac_map


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    https://stackoverflow.com/questions/50834820/get-the-label-mappings-from-label-encoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res


def encode_labels(device_list):
    """
    Encodes the list of devices into integer values
    to enable the ma
    """
    # Conver list of devices to numpy array
    device_numpy = np.array(device_list)

    # Setup the labelencoder
    labelencoder = LabelEncoder()
    encoded_labels = labelencoder.fit_transform(device_numpy)

    # Get label mapping
    label_mapping = get_integer_mapping(labelencoder)
    print(label_mapping)
    return labelencoder, label_mapping


def convert_to_tensor(features, labels):
    # Convert features to tensor format
    tensor_features = torch.from_numpy(features.astype(float))

    # Convert labels to tensor format
    tensor_labels = torch.from_numpy(labels)

    return tensor_features, tensor_labels



def make_dataset_iterable(data_x, data_y, device):
    """
    Use the Tensor dataloader to convert data batches
    """
    batch_size = 100

    tensor_dataset = TensorDataset(data_x, data_y)

    if device == "cuda":
        dataloader = DataLoader(dataset=tensor_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=NUM_PROCS,
                                    pin_memory_device=device)
    
    else:
        dataloader = DataLoader(dataset=tensor_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=NUM_PROCS)

    return dataloader
