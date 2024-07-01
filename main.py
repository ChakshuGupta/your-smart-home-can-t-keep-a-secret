import json
import logging
import os
import sys
import torch
import yaml

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from src.traffic_process import preprocess_traffic
from src.train_test_model import train_lstm_model, test_lstm_model
from src.util import encode_labels, load_device_file


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
    
    # if "device-file" not in config:
    #     error = True
    # else:
    #     if not os.path.exists(config["device-file"]):
    #         error = True
    #     elif not os.path.isfile(config["device-file"]):
    #         error = True
    #     elif not config["device-file"].endswith(".txt"):
    #         error = True
    
    return error


def get_pcap_list(dataset_dir):
    """
    Get the list of pcap files in the directory
    """
    pcap_files = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".pcap") or file.endswith(".pcapng"):
                pcap_files.append(os.path.join(root, file))
    pcap_files.sort()
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
    
    ####### Set up Logger ########

    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_filepath = os.path.join("logs" , "log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler(log_filepath)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    ##############################

    # Verify the validity of the config file
    error = verify_config(config)
    if error:
        logger.error("ERROR! Invalid config file!")
        exit(1)

    # Select CUDA option if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    
    logger.info("Using the device: {}".format(device))
    # Load the file mapping mac addresses to devices
    # device_mac_map = load_device_file(config["device-file"])
    # Get the list of values from the map i.e the list of devices
    # device_list = list(device_mac_map.values())
    device_list = next(os.walk(config["dataset-path"]["train"]))[2]
    logger.info("Device-mac_address mapping file loaded in memory.")
    # Get the label encoder and label mapping by encoding the device list into integers
    labelencoder, label_mapping = encode_labels(device_list)

    # Get the base path for the pickle files
    dataset_base_path = os.path.join(os.getcwd(), "data" , os.path.basename(config["dataset-path"]["train"]))
    if not os.path.exists("data"):
        os.makedirs("data")

    if config["dataset-path"]["train"] == config["dataset-path"]["test"]:
        cross_validation = True
        logger.info("Cross validation flag is active. Hence, training and testing on the same dataset: {}".format(config["dataset-path"]["train"]))
    else:
        cross_validation = False
        logger.info("No cross validation. Training and testing on different datasets: Train: {}, Test: {}".format(config["dataset-path"]["train"], config["dataset-path"]["test"]))

    if cross_validation:
        # Get the list of pcaps from the dataset dir
        dataset_pcap_list = get_pcap_list(config["dataset-path"]["train"])
        logger.info("Retreived the list of pcaps from the dataset's directory.")
        # Preprocess the pcap files to get the features and the labels
        dataset_x, dataset_y = preprocess_traffic(dataset_pcap_list, dataset_base_path)
        logger.info("Finished loading and preprocessing the data.")

        # Declare the lists
        y_true_all = []
        y_pred_all = []
        # Declare the stratified k fold object
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        idx = 0
        # Loop through the different folds
        for train_index, test_index in skf.split(dataset_x, dataset_y):
            logger.info("Starting Fold number: {}".format(str(idx)))
            # split the dataset into train and test dataset using the indices
            x_train = dataset_x[train_index]
            y_train = dataset_y[train_index]
            x_test = dataset_x[test_index]
            y_test = dataset_y[test_index]
            
            # Get the encoded labels for the training dataset
            y_train = labelencoder.transform(y_train.ravel())
            model_path = dataset_base_path + "-model-" + str(idx) + ".sav"
            # Train the LSTM model
            model = train_lstm_model(x_train, y_train, label_mapping, model_path, bidirectional=False, device=device)
            logger.info("Finished training the model for the fold number: {}".format(str(idx)))
            logger.info("Model is saved in the location: {}".format(model_path))
            # Get the encoded labels for the testing dataset
            y_test = labelencoder.transform(y_test.ravel())
            # Test the Generated model
            y_true, y_pred = test_lstm_model(model, x_test, y_test, labelencoder, device=device)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            # increment index
            idx += 1
        
        # Print the classification report
        report = classification_report(
                            y_true = y_true_all,
                            y_pred = y_pred_all,
                            digits = 4,
                            zero_division = 0,
                            output_dict=True
                        )
        
        report_file_path = dataset_base_path + "-report.json"
        report_file = open(report_file_path, "w")
        report_file.write(json.dumps(report, indent=2))
        report_file.close()
        logger.info("Completed testing the model and the report is now saved in the json file : {}".format(report_file_path))


    else:
    
        # Prepare the training data
        train_pcap_list = get_pcap_list(config["dataset-path"]["train"])
        logger.info("Retreived the list of pcaps from the training dataset's directory.")
        # Preprocess the traffic and get the features from the packets
        x_train, y_train = preprocess_traffic(device_mac_map, train_pcap_list, dataset_base_path)
        logger.info("Finished loading and preprocessing the training data.")
        # Encode the training labels using the labelencoder
        y_train = labelencoder.transform(y_train.ravel())

        model_path = dataset_base_path + "-model.sav"
        # Train the LSTM model
        model = train_lstm_model(x_train, y_train, label_mapping, model_path, bidirectional=False, device=device)
        logger.info("Finished training the model!")
        logger.info("Model is saved in the location: {}".format(model_path))
        # Prepare the testing data
        test_pcap_list = get_pcap_list(config["dataset-path"]["test"])
        logger.info("Retreived the list of pcaps from the testing dataset's directory.")
        # Get the base path of the pickle file using the testing dataset path
        test_base_path = os.path.join(os.getcwd(), os.path.basename(config["dataset-path"]["test"]))
        # Preprocess the test traffic and get the features from the packets
        x_test, y_test = preprocess_traffic(device_mac_map, test_pcap_list, test_base_path)
        logger.info("Finished loading and preprocessing the testing data.")
        # Encode the test labels using the labelencoder
        y_test = labelencoder.transform(y_test.ravel())
        logger.info("C0mpleted testing the model and the report is now saved in the json file.")
        # Test the Generated model
        y_true, y_pred = test_lstm_model(model, x_test, y_test, labelencoder, device=device)

        # Print the classification report
        report = classification_report(
                            y_true = y_true,
                            y_pred = y_pred,
                            digits = 4,
                            zero_division = 0,
                            output_dict=True
                        )

        report_file_path = dataset_base_path +  "-report.json"
        report_file = open(report_file_path, "w")
        report_file.write(json.dumps(report, indent=2))
        report_file.close()
        logger.info("C0mpleted testing the model and the report is now saved in the json file : {}".format(report_file_path))

    print(label_mapping)