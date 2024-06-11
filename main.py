import os
import sys
import torch
import yaml

from src.traffic_process import preprocess_traffic
from src.util import convert_to_tensor

def verify_config(config):
    """
    Verify the entries in the config file pass the basic sanity checks
    """
    error = False

    if "dataset-path" not in config:
        error = True
    else:
        if not os.path.exists(config["dataset-path"]):
            error = True
        elif not os.path.isdir(config["dataset-path"]):
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
    
    config_file = sys.argv[1]
    
    if not config_file.endswith("yml") and not config_file.endswith("yaml"):
        print("ERROR! The config file is not a YAML file.")
        exit(1)
    if not os.path.exists(config_file):
        print("ERROR! The path to the config file does not exist.")
        exit(1)
    
    with open(config_file, "r") as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)
    
    error = verify_config(config)
    
    if error:
        print("ERROR! Invalid config file!")
        exit(1)

    device_mac_map = load_device_file(config["device-file"])

    pcap_list = get_pcap_list(config["dataset-path"])

    pickle_path = os.path.join(os.getcwd(), os.path.basename(config["dataset-path"]))

    # Preprocess the traffic and get the fingerprints from the packets
    train_features, train_labels = preprocess_traffic(device_mac_map, pcap_list, pickle_path)

    # # Generate the base model for comparison
    # basemodel = BaseModel(features)
    # model_file = pickle_path + "-random-forest_model.sav"
    # if not os.path.exists(model_file):
    #     basemodel.build_model(labels)
    #     basemodel.save_model(model_file)
    
    train_features["dport"] = train_features["dport"].astype(int)
    train_features["frame_len"] = train_features["frame_len"].astype(int)
    train_features["protocol"] = train_features["protocol"].astype(int)

    # print(labels.columns)

    train_x, train_y, label_mapping = convert_to_tensor(train_features, train_labels)
    
    print(train_x.shape, train_y.shape)
    print(label_mapping)