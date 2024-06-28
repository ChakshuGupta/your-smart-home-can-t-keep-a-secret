import numpy as np
import os
import pandas as pd
import pickle
import time

from multiprocessing import Process, Manager

from scapy.all import *
from scapy.layers.tls.record import TLS # Import this to ensure TLS layers are read by rdpcap
from subprocess import Popen, PIPE

from src.feature_extractor import extract_features


NUM_PROCS = 4


def process_pcap_tshark(file_list, mac_addrs, dataset):
    """
    Read and process the pcap file using command line tshark.
    Extract the features from the file and return them.
    """
    for file in file_list:
        print("Reading file: ", file)
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
            feature_vector = extract_features(packet, last_time, use_tshark=True)

            # If the src or dst MAC address exists in the mapping
            # add the corresponding device name in the label
            src_mac = packet[3]
            dst_mac = packet[4]

            # Add to the list only if the src or dst mac address is there in the
            # mapping
            if src_mac in mac_addrs:
                # append the fingerprint in the features list
                dataset.append((float(packet[1]), feature_vector.__dict__, mac_addrs[src_mac]))
            elif dst_mac in mac_addrs:
                # append the fingerprint in the features list
                dataset.append((float(packet[1]), feature_vector.__dict__, mac_addrs[dst_mac]))
            
            last_packet = packet
            index += 1
    

def process_pcap_scapy(file_list, mac_addrs, dataset):
    """
    Read and process the pcap file using scapy library.
    Extract the features from the file and return them.
    """
    for file in file_list:
        print("Reading file: ", file)
        packets = rdpcap(file)

        for packet in packets:
            for index, packet in enumerate(packets):
                if index == 0:
                    last_time = 0
                else:
                    last_time = float(packets[index-1].time)
                # Extract fingerprint for the packet
                feature_vector = extract_features(packet, last_time, use_tshark=False)
                print(feature_vector)
                # If the src or dst MAC address exists in the mapping
                # add the corresponding device name in the label
                src_mac = packet["Ether"].src
                dst_mac = packet["Ether"].dst

                # Add to the list only if the src or dst mac address is there in the
                # mapping
                if src_mac in mac_addrs:
                    # append the fingerprint in the features list
                    dataset.append((packet.time, feature_vector.__dict__, mac_addrs[src_mac]))
                elif dst_mac in mac_addrs:
                    # append the fingerprint in the features list
                    dataset.append((packet.time, feature_vector.__dict__, mac_addrs[dst_mac]))
        

def preprocess_traffic(mac_addrs, pcap_list, pickle_path):
    """
    Preprocess the traffic and extract the features from the traffic
    """
    # Initialize the empty lists
    dataset = Manager().list()

    # Get the filepaths for the features and labels pickle files
    features_filepath = pickle_path + "-" + "features.pickle"
    labels_filepath = pickle_path + "-" + "labels.pickle"

    # If the files already exist, load them
    if os.path.exists(features_filepath) and os.path.exists(labels_filepath):
        print("Loading the pickle files: {} and {}".format(features_filepath, labels_filepath))
        dataset_x = pickle.load(open(features_filepath, "rb"))
        dataset_y = pickle.load(open(labels_filepath, "rb"))
       
        # return the loaded values
        return dataset_x, dataset_y

    use_tshark = True
    print("Pickle files do not exist. Reading the pcap files...")
    start = time.perf_counter()
    chunk_size = int(len(pcap_list)/NUM_PROCS)

    pcap_list_split = []

    for idx in range(0, len(pcap_list), chunk_size):
        pcap_list_split.append(pcap_list[idx:idx+chunk_size])
    print(pcap_list_split)
    # If the files do not exist, it will continue here.
    processes = []
    for chunk in pcap_list_split:
        if use_tshark:
            p = Process(target=process_pcap_tshark, args=(chunk, mac_addrs, dataset))
        else:
            p = Process(target=process_pcap_scapy, args=(chunk, mac_addrs, dataset))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    end = time.perf_counter()
    print(f'It took {end-start:.2f} second(s) to finish')

    # Sort the data using the epoch time
    dataset = list(sorted(dataset, key=lambda tup: tup[0]))
    # Split the tuple values into epoch times, features and labels
    _, features, labels = zip(*dataset)
    # convert the lists to dataframe
    df_features = pd.DataFrame.from_dict(features)

    # Split the protocols list into separate columns
    df_features[["protocol1", "protocol2", "protocol3", "protocol4", "protocol5", "protocol6", "protocol7"]] =\
        pd.DataFrame(df_features.protocol.to_list(), index=df_features.index)
    # Delete the original protocol column from the dataframe
    del df_features["protocol"]

    # Convert labels list to dataframe
    df_labels = pd.DataFrame(labels)

    dataset_x, dataset_y = get_sliding_windows(df_features, df_labels, 20)

    print("Saving the extracted features into pickle files.")
    # Save the dataframes to pickle files    
    pickle.dump(dataset_x, open(features_filepath, "wb"))
    pickle.dump(dataset_y, open(labels_filepath, "wb"))

    print( "Length of the features list and labels list: ", len(dataset_x), len(dataset_y))
    return dataset_x, dataset_y


def split_traffic(data_chunk, window_size, dataset_x, dataset_y):
    # For each label, get the data for that label
    for label in data_chunk:
        print(label)
        data = data_chunk[label]

        if data.shape[0] <= window_size:
            num_null_rows = data.shape[0] + window_size
            for i in range(data.shape[0]-1, num_null_rows):
                pd.concat([data, pd.Series()], ignore_index=False)

        for idx in range(data.shape[0] - window_size):
            traffic_window = data.iloc[ idx:idx + window_size, :]
            del traffic_window["label"]
            dataset_x.append(traffic_window.values.tolist())
            dataset_y.append(label)


def get_sliding_windows(df_features, df_labels, window_size):
    """
    Split the traffic
    """
    # Add the labels as a column to the features dataframe
    df_features["label"] = df_labels
    # Declare empty lists for the trafic windows
    dataset_x = Manager().list()
    dataset_y = Manager().list()
    # Get the unique labels in the dataframe
    unique_labels = df_features["label"].unique()
    labelwise_data = {}
    for label in unique_labels:
        labelwise_data[label] = df_features.loc[df_features["label"] == label]
    
    chunk_size = int(len(unique_labels)/NUM_PROCS)

    data_split = []
    for idx in range(0, len(unique_labels), chunk_size):
        split_labels = unique_labels[idx: idx+chunk_size]
        temp = {}
        for label in split_labels:
            temp[label] = labelwise_data[label]
        data_split.append(temp)
    
    processes = []
    for idx, data in enumerate(data_split):
        p = Process(target=split_traffic, args=(data, window_size, dataset_x, dataset_y))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()    
   
    return np.array(dataset_x, dtype=object), np.array(dataset_y)