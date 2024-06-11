import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


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



def convert_to_tensor(features, labels):
    # Convert features to tensor format
    tensor_features = torch.from_numpy(features.to_numpy())

    # Encode the labels to interger values
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels.values.ravel())
    # Get the labels mapped to integer values
    label_mapping = get_integer_mapping(labelencoder)
    # Convert labels to tensor format
    tensor_labels = torch.from_numpy(labels)

    return tensor_features, tensor_labels, label_mapping



def make_dataset_iterable(data_x, data_y):
    """
    Use the Tensor dataloader to convert data batches
    """
    batch_size = 100

    tensor_dataset = TensorDataset (data_x, data_y)

    dataloader = DataLoader(dataset=tensor_dataset, 
                                batch_size=batch_size, 
                                shuffle=False)

    return dataloader
