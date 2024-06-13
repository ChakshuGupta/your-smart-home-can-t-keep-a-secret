import numpy as np
import os
import pandas as pd
import pickle
import pyshark
from src.feature_extractor import extract_features
from subprocess import Popen, PIPE

def preprocess_traffic(mac_addrs, pcap_list, pickle_path):
    """
    Preprocess the traffic and extract the features from the traffic
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
        # packets = pyshark.FileCapture(file)

        command = ["tshark", "-r", file,
                   "-Tfields",
                   "-e", "frame.len",
                   "-e", "frame.time_epoch",
                   "-e", "frame.protocols",
                   "-e", "eth.src",
                   "-e", "eth.dst",
                   "-e", "ip.src",
                   "-e", "ip.dst",
                   "-e", "ipv6.src",
                   "-e", "ipv6.dst",
                   "-e", "tcp.dstport",
                   "-e", "udp.dstport"
                   ]

        # Call Tshark on packets
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        # Get output. Give warning message if any
        out, err = process.communicate()
        if err:
            print("Error reading file: '{}'".format(err.decode('utf-8')))
        

        index = 0
        last_packet = None
        for packet in filter(None, out.decode('utf-8').split('\n')):
            packet = np.array(packet.split())
            print(packet)
            if index == 0:
                last_time = 0
            else:
                last_time = float(last_packet[1])
            
            if len(packet) < 8:
                continue
            # Extract fingerprint for the packet
            feature_vector = extract_features(packet, last_time)

            # If the src or dst MAC address exists in the mapping
            # add the corresponding device name in the label
            src_mac = packet[3]
            dst_mac = packet[4]

            # Add to the list only if the src or dst mac address is there in the
            # mapping
            if src_mac in mac_addrs:
                # append the fingerprint in the features list
                features.append(feature_vector.__dict__)
                labels.append(mac_addrs[src_mac])
            elif dst_mac in mac_addrs:
                # append the fingerprint in the features list
                features.append(feature_vector.__dict__)
                labels.append(mac_addrs[dst_mac])
            
            last_packet = packet
            index += 1

        # # Close the capture file and clear the data
        # packets.close()
        # packets.clear()

    # convert the lists to dataframe
    df_features = pd.DataFrame.from_dict(features)
    
    # Split the protocols list into separate columns
    df_features[["protocol1", "protocol2", "protocol3", "protocol4", "protocol5", "protocol6", "protocol7"]] =\
        pd.DataFrame(df_features.protocol.to_list(), index=df_features.index)
    # Delete the original protocol column from the dataframe
    del df_features["protocol"]

    # Convert labels list to dataframe
    df_labels = pd.DataFrame(labels)

    print("Saving the extracted features into pickle files.")
    # Save the dataframes to pickle files    
    pickle.dump(df_features, open(features_filepath, "wb"))
    pickle.dump(df_labels, open(labels_filepath, "wb"))

    print( "Length of the features list and labels list: ", len(features), len(labels))
    return df_features, df_labels