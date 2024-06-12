import os
import sys
import yaml

from sklearn.model_selection import StratifiedKFold

from src.traffic_process import preprocess_traffic
from src.train_test_model import train_lstm_model, test_lstm_model
from src.util import encode_labels


def verify_config(config):
    """
    Verify the entries in the config file pass the basic sanity checks
    """
    error = False

    if "dataset-path" not in config:
        error = True
    else:
        if not os.path.exists(config["dataset-path"]["train"]):
            error = True
        elif not os.path.isdir(config["dataset-path"]["train"]):
            error = True

        if not os.path.exists(config["dataset-path"]["test"]):
            error = True
        elif not os.path.isdir(config["dataset-path"]["test"]):
            error = True
    
    if "device-file" not in config:
        error = True
    else:
        if not os.path.exists(config["device-file"]):
            error = True
        elif not os.path.isfile(config["device-file"]):
            error = True
        elif not config["device-file"].endswith(".txt"):
            error = True
    
    return error


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


def get_pcap_list(dataset_dir):
    """
    Get the list of pcap files in the directory
    """
    pcap_files = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".pcap") or file.endswith(".pcapng"):
                pcap_files.append(os.path.join(root, file))

    return pcap_files


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR! THe script requires the path to the config file as argument")
    
    # Get the path to the config file from the argument
    config_file = sys.argv[1]
    # Validate the file path
    if not config_file.endswith("yml") and not config_file.endswith("yaml"):
        print("ERROR! The config file is not a YAML file.")
        exit(1)
    if not os.path.exists(config_file):
        print("ERROR! The path to the config file does not exist.")
        exit(1)
    # Load the config values
    with open(config_file, "r") as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)
    
    # Verify the validity of the config file
    error = verify_config(config)
    if error:
        print("ERROR! Invalid config file!")
        exit(1)
    
    # Load the file mapping mac addresses to devices
    device_mac_map = load_device_file(config["device-file"])
    # Get the list of values from the map i.e the list of devices
    device_list = list(device_mac_map.values())
    device_list.append("local")
    # Get the label encoder and label mapping by encoding the device list into integers
    labelencoder, label_mapping = encode_labels(device_list)

    if config["dataset-path"]["train"] == config["dataset-path"]["test"]:
        cross_validation = True
    else:
        cross_validation = False

    if cross_validation:
        # Get the list of pcaps from the dataset dir
        dataset_pcap_list = get_pcap_list(config["dataset-path"]["train"])
        # Get the base path for the pickle files
        dataset_pickle_path = os.path.join(os.getcwd(), os.path.basename(config["dataset-path"]["train"]))
        # Preprocess the pcap files to get the features and the labels
        dataset_x, dataset_y = preprocess_traffic(device_mac_map, dataset_pcap_list, dataset_pickle_path)
        # Declare the stratified k fold object
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        # Loop through the different folds
        for train_index, test_index in skf.split(dataset_x, dataset_y):
            # split the dataset into train and test dataset using the indices
            x_train = dataset_x.iloc[train_index]
            y_train = dataset_y.iloc[train_index]
            x_test = dataset_x.iloc[test_index]
            y_test = dataset_y.iloc[test_index]
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            
            # Get the encoded labels for the training dataset
            y_train = labelencoder.transform(y_train.values.ravel())
            # Train the LSTM model
            model = train_lstm_model(x_train, y_train, label_mapping)
            
            # Get the encoded labels for the testing dataset
            y_test = labelencoder.transform(y_test.values.ravel())
            # Test the Generated model
            test_lstm_model(model, x_test, y_test)


    else:
    
        # Prepare the training data
        train_pcap_list = get_pcap_list(config["dataset-path"]["train"])
        # Get the base path of the pickle file using the training dataset path
        train_pickle_path = os.path.join(os.getcwd(), os.path.basename(config["dataset-path"]["train"]))

        # Preprocess the traffic and get the features from the packets
        x_train, y_train = preprocess_traffic(device_mac_map, train_pcap_list, train_pickle_path)
        # Encode the training labels using the labelencoder
        y_train = labelencoder.transform(y_train.values.ravel())


        # Train the LSTM model
        model = train_lstm_model(x_train, y_train, label_mapping)

        # Prepare the testing data
        test_pcap_list = get_pcap_list(config["dataset-path"]["test"])
        # Get the base path of the pickle file using the testing dataset path
        test_pickle_path = os.path.join(os.getcwd(), os.path.basename(config["dataset-path"]["test"]))
        # Preprocess the test traffic and get the features from the packets
        x_test, y_test = preprocess_traffic(device_mac_map, test_pcap_list, test_pickle_path)
        # Encode the test labels using the labelencoder
        y_test = labelencoder.transform(y_test.values.ravel())

        # Test the Generated model
        test_lstm_model(model, x_test, y_test)