import numpy as np
import os
import pandas as pd
import pickle
import time

from functools import partial
# from multiprocessing import Process, Manager
from multiprocessing.pool import Pool

from scapy.all import *
from scapy.layers.tls.record import TLS # Import this to ensure TLS layers are read by rdpcap
from subprocess import Popen, PIPE

from src.feature_extractor import extract_features


NUM_PROCS = 3


def process_pcap_tshark(mac_addrs, file):
    """
    Read and process the pcap file using command line tshark.
    Extract the features from the file and return them.
    """
    dataset = []
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
    return dataset
    

def process_pcap_scapy(mac_addrs, file):
    """
    Read and process the pcap file using scapy library.
    Extract the features from the file and return them.
    """
    dataset = []

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
    return dataset
        

def preprocess_traffic(mac_addrs, pcap_list, pickle_path):
    """
    Preprocess the traffic and extract the features from the traffic
    """
    # Get the filepaths for the partial features and labels pickle files (without sliding windows)
    features_part_filepath = pickle_path + "-" + "features-part.pickle"
    labels_part_filepath = pickle_path + "-" + "labels-part.pickle"
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

    if os.path.exists(features_part_filepath) and os.path.exists(labels_part_filepath):
        print("Loading the pickle files: {} and {}".format(features_part_filepath, labels_part_filepath))
        df_features = pickle.load(open(features_part_filepath, "rb"))
        df_labels = pickle.load(open(labels_part_filepath, "rb"))
    
    else:

        use_tshark = True
        print("Pickle files do not exist. Reading the pcap files...")
        start = time.perf_counter()

        # pcap_list_split = [ [] for _ in range(NUM_PROCS) ]

        # for idx in range(0, len(pcap_list)):
        #     pcap_list_split[idx % NUM_PROCS].append(pcap_list[idx])
        
        # print(len(pcap_list_split))

        dataset = []

        # If the files do not exist, it will continue here.
        pool = Pool(processes=NUM_PROCS)

        if use_tshark:
            func = partial(process_pcap_tshark, mac_addrs)
        else:
            func = partial(process_pcap_scapy, mac_addrs)

        for result in pool.map(func, pcap_list):
            dataset.extend(result)
        
        end = time.perf_counter()
        print(f'It took {end-start:.2f} second(s) to finish')

        pool.close()
        pool.join()
        
        print(f"Number of fingerprints: {len(dataset)}")

        # Sort the data using the epoch time
        sorted_dataset = list(sorted(dataset, key=lambda tup: tup[0]))
        # Split the tuple values into epoch times, features and labels
        _, features, labels = zip(*sorted_dataset)
        # convert the lists to dataframe
        df_features = pd.DataFrame.from_dict(features)

        # Split the protocols list into separate columns
        df_features[["protocol1", "protocol2", "protocol3", "protocol4", "protocol5", "protocol6", "protocol7"]] =\
            pd.DataFrame(df_features.protocol.to_list(), index=df_features.index)
        # Delete the original protocol column from the dataframe
        del df_features["protocol"]

        # Convert labels list to dataframe
        df_labels = pd.DataFrame(labels)

        pickle.dump(df_features, open(features_part_filepath, "wb"))
        pickle.dump(df_labels, open(labels_part_filepath, "wb"))


    dataset = get_sliding_windows(df_features, df_labels, 20)
    np.random.shuffle(dataset)
    
    print(dataset)
    dataset_x, dataset_y = zip(*dataset)
    dataset_x = np.array(dataset_x, dtype=object)
    dataset_y = np.array(dataset_y, dtype=object)

    print("Saving the extracted features into pickle files.")
    # Save the dataframes to pickle files    
    pickle.dump(dataset_x, open(features_filepath, "wb"))
    pickle.dump(dataset_y, open(labels_filepath, "wb"))

    print( "Length of the features list and labels list: ", len(dataset_x), len(dataset_y))
    return dataset_x, dataset_y


def split_traffic(window_size, data_chunk):
    dataset = []

    label = list(data_chunk.keys())[0]   
    
    data = data_chunk[label]
    del data["label"]

    if data.shape[0] <= window_size:
        num_null_rows = data.shape[0] + window_size
        for i in range(data.shape[0]-1, num_null_rows):
            pd.concat([data, pd.Series()], ignore_index=False)
    
    print(label, data.shape[0] - window_size)
    
    for idx in range(data.shape[0] - window_size):
        traffic_window = data.iloc[ idx:idx + window_size, :]
        dataset.append((traffic_window.values.tolist(), label))

    print(dataset)

    return dataset


def get_sliding_windows(df_features, df_labels, window_size):
    """
    Split the traffic
    """
    # manager = Manager()
    # Add the labels as a column to the features dataframe
    df_features["label"] = df_labels
    # Declare empty lists for the trafic windows
    # dataset = manager.list()
    # Get the unique labels in the dataframe
    unique_labels = df_features["label"].unique()
    labelwise_data = []
    for label in unique_labels:
        labelwise_data.append({label: df_features.loc[df_features["label"] == label]})
    
    # data_split = [ {} for _ in range(NUM_PROCS) ]
    # for idx in range(0, len(unique_labels)):
    #     data_split[idx % NUM_PROCS][unique_labels[idx]] = labelwise_data[unique_labels[idx]]
    
    # print("Length of split data", len(data_split))
    dataset = []
    pool = Pool(processes=NUM_PROCS)

    func = partial(split_traffic, window_size)

    for result in pool.map(func, labelwise_data):
        dataset.extend(result)

    pool.close()
    pool.join()

    print(dataset)

    # processes = []
    # for idx, data in enumerate(data_split):
    #     process = Process(target=split_traffic, args=(data, window_size))
    #     processes.append(process)
    #     process.start()
    #     print(process)
    
    # for process in processes:
    #     print(process)
    #     process.join()    
   
    return np.array(dataset, dtype=object)