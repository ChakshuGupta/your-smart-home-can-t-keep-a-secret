import os
import pandas as pd
import pickle
import pyshark
from src.fingerprint_extractor import extract_features


def preprocess_traffic(mac_addrs, pcap_list, pickle_path):
    """
    """
    # Initialize the empty lists
    features = []
    labels = []

    # Get the filepaths for the features and labels pickle files
    features_filepath = pickle_path + "-" + "features.pickle"
    labels_filepath = pickle_path + "-" + "labels.pickle"

    # If the files already exist, load them
    if os.path.exists(features_filepath) and os.path.exists(labels_filepath):
        print("Loading the pickle files: {} and {}".format(features_filepath, labels_filepath))
        df_features = pickle.load(open(features_filepath, "rb"))
        df_labels = pickle.load(open(labels_filepath, "rb"))
        
        # return the loaded values
        return df_features, df_labels

    print("Pickle files do not exist. Reading the pcap files...")
    # If the files do not exist, it will continue here.
    for file in pcap_list:
        print("Reading file: ", file)
        packets = pyshark.FileCapture(file)
        for index, packet in enumerate(packets):
            if index == 0:
                last_time = 0
            else:
                last_time = float(packets[index-1].frame_info.time_epoch)
            # Extract fingerprint for the packet
            fingerprint = extract_features(packet, last_time)

            if fingerprint.is_none():
                continue
            
            # append the fingerprint in the features list
            features.append(fingerprint.__dict__)

            # If the src or dst MAC address exists in the mapping
            # add the corresponding device name in the label
            src_mac = packet.eth.src
            dst_mac = packet.eth.dst

            if src_mac in mac_addrs:
                labels.append(mac_addrs[src_mac])
            elif dst_mac in mac_addrs:
                labels.append(mac_addrs[dst_mac])
            else:
                # if the mac addr does not exist in the mapping append "local"
                # to the labels
                labels.append("local")
        
        # Close the capture file and clear the data
        packets.close()
        packets.clear()
    print(features)

    # convert the lists to dataframe
    df_features = pd.DataFrame.from_dict(features)
    df_labels = pd.DataFrame(labels)

    print("Saving the extracted features into pickle files.")
    # Save the dataframes to pickle files    
    pickle.dump(df_features, open(features_filepath, "wb"))
    pickle.dump(df_labels, open(labels_filepath, "wb"))

    print( "Length of the features list and labels list: ", len(features), len(labels))
    return df_features, df_labels