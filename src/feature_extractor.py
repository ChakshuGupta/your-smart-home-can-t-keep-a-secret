import ipaddress

from scapy.all import *
from scapy.layers.tls.record import TLS

from src.object.feature_vector import FeatureVector
  

def get_protocol_list(packet, use_tshark):
    """
    Generate the protocols string using the method described in the paper
    """
    # default values of the protocols
    ip, tcp, udp, tls, http, dns, other = 0,0,0,0,0,0,0
    if use_tshark:
        protocols = packet[2]
        protocol_list = protocols.split(":")
        for proto in protocol_list:
            if proto == "eth":
                continue
            elif proto == "ethertype":
                continue
            elif proto == "ip":
                ip = 1
            elif proto == "tcp":
                tcp = 1
            elif proto == "udp":
                udp = 1
            elif proto == "tls" or proto == "ssl":
                tls = 1
            elif proto == "http":
                http = 1
            elif proto == "dns":
                dns = 1
            else:
                other = 1
    else:
        layers = packet.layers()
        layers.remove(Ether)
        if IP in layers:
            ip = 1
            layers.remove(IP)
        if TCP in layers:
            tcp = 1
            layers.remove(TCP)
            if packet.sport == 80 or packet.dport == 80:
                http = 1
        if UDP in layers:
            udp = 1
            layers.remove(UDP)
            if packet.sport == 80 or packet.dport == 80:
                http = 1
        if DNS in layers:
            dns = 1
            layers.remove(DNS)
        if Raw in layers:
            pkt_data = packet[Raw].load
            if pkt_data is not None and "TLS" in TLS(pkt_data):
                tls = 1
            layers.remove(Raw)
        if len(layers)>1:
            other = 1


    proto_list = [ip, tcp, udp, tls, http, dns, other]

    return proto_list


def get_direction(packet, use_tshark):
    """
    Get the direction of the traffic: inbound (0) or outbound (1)
    """
    # If using tshark
    if use_tshark:
        # if "IPV6" in str(packet.layers):
        if "," in packet[5]:
            packet[5] = packet[5].split(",")[0]
        
        if "," in packet[6]:
            packet[6] = packet[6].split(",")[0]

        src = ipaddress.ip_address(packet[5])
        dst = ipaddress.ip_address(packet[6])

        if src.is_global:
            return 0
        elif dst.is_global:
            return 1
        else:
            return 0
    
    # If using scapy:
    else:
        ip_layer = ipaddress.ip_address("0.0.0.0")
        if "IPV6" in packet:
            ip_layer = ipaddress.ip_address(packet["IPv6"])

        elif "IP" in packet:
            ip_layer = ipaddress.ip_address(packet["IP"])

        src = ipaddress.ip_address(ip_layer.src)   
        dst = ipaddress.ip_address(ip_layer.dst)

        if src.is_global:
            return 0
        elif dst.is_global:
            return 1
        else:
            return 0


def get_dport(packet):
    """
    Get the destination port from the transport layer
    """
    if "TCP" in packet or "UDP" in packet:
        return int(packet.dport)
    else:
        return None
    
    # This is for pyshark
    # transport_layer = packet.transport_layer
    # if transport_layer is None:
    #     return None
    
    # return int(packet[packet.transport_layer].dstport)


def extract_features(packet, last_time, use_tshark):
    """
    Extract the features required from the packet and store it
    in the object of the fingerprint class.
    """
    feature_vector = FeatureVector()

    if use_tshark:
        feature_vector.frame_len = int(packet[0])
        feature_vector.time_interval = float(packet[1]) - last_time
        feature_vector.protocol = get_protocol_list(packet, use_tshark)
        feature_vector.direction = get_direction(packet, use_tshark)
        feature_vector.dport = int(packet[7])
    
    else:
        feature_vector.frame_len = len(packet)
        feature_vector.time_interval = float(packet.time) - last_time
        feature_vector.protocol = get_protocol_list(packet, use_tshark)
        feature_vector.direction = get_direction(packet, use_tshark)
        feature_vector.dport = get_dport(packet)

    return feature_vector