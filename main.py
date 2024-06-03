import os
import sys
import yaml

from src.preprocessor import preprocess_traffic
from src.util import verify_config



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
    print(device_mac_map)

    pcap_list = get_pcap_list(config["dataset-path"])
    print(pcap_list)

    # Preprocess the traffic and get the fingerprints from the packets
    preprocess_traffic(device_mac_map, pcap_list)