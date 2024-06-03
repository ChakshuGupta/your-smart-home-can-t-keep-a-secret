import pandas as pd
import pyshark
from src.fingerprint_extractor import extract_features


def preprocess_traffic(mac_addrs, pcap_list):
    features = []
    for file in pcap_list:
        packets = pyshark.FileCapture(file)
        for index, packet in enumerate(packets):
            if index == 0:
                last_time = 0
            else:
                last_time = float(packets[index-1].frame_info.time_epoch)
            fingerprint = extract_features(packet, last_time)
            features.append(fingerprint.__dict__)
        
        packets.close()
        packets.clear()

    df = pd.DataFrame(d=features)