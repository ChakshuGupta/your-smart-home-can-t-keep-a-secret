import pandas as pd
import pyshark
from src.fingerprint_extractor import extract_features


def preprocess_traffic(mac_addrs, pcap_list):
    """
    """
    features = []
    labels = []
    for file in pcap_list:
        packets = pyshark.FileCapture(file)
        for index, packet in enumerate(packets):
            if index == 0:
                last_time = 0
            else:
                last_time = float(packets[index-1].frame_info.time_epoch)
            fingerprint = extract_features(packet, last_time)

            if fingerprint.is_none():
                continue

            features.append(fingerprint.__dict__)

            src_mac = packet.eth.src
            dst_mac = packet.eth.dst

            if src_mac in mac_addrs:
                labels.append(mac_addrs[src_mac])
            elif dst_mac in mac_addrs:
                labels.append(mac_addrs[dst_mac])
            else:
                labels.append("local")
        
        packets.close()
        packets.clear()

    df_features = pd.DataFrame.from_dict(features)
    df_labels = pd.DataFrame(labels)
    print( "Length of the features list and labels list: ", len(features), len(labels))