import ipaddress
from src.object.feature_vector import FeatureVector
  

def get_protocol_string(protocols):
    """
    Generate the protocols string using the method described in the paper
    """
    # default values of the protocols
    ip, tcp, udp, tls, http, dns, other = "0", "0", "0", "0", "0", "0", "0"
    protocol_list = protocols.split(":")
    for proto in protocol_list:
        if proto == "eth":
            continue
        elif proto == "ethertype":
            continue
        elif proto == "ip":
            ip = "1"
        elif proto == "tcp":
            tcp = "1"
        elif proto == "udp":
            udp = "1"
        elif proto == "tls" or proto == "ssl":
            tls = "1"
        elif proto == "http":
            http = "1"
        elif proto == "dns":
            dns = "1"
        else:
            other = "1"

    proto_str = ip + tcp + udp + tls + http + dns + other

    return proto_str


def get_direction(packet):
    """
    Get the direction of the traffic: inbound (0) or outbound (1)
    """
    if "IPV6" in str(packet.layers):
        src = ipaddress.ip_address(packet.ipv6.src)
        dst = ipaddress.ip_address(packet.ipv6.dst)

        if src.is_global:
            return 0
        elif dst.is_global:
            return 1
        else:
            return 0

    elif "IP" in str(packet.layers):
        src = ipaddress.ip_address(packet.ip.src)
        dst = ipaddress.ip_address(packet.ip.dst)

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
    transport_layer = packet.transport_layer
    if transport_layer is None:
        return None
    
    return int(packet[packet.transport_layer].dstport)


def extract_features(packet, last_time):
    """
    Extract the features required from the packet and store it
    in the object of the fingerprint class.
    """
    feature_vector = FeatureVector()

    feature_vector.frame_len = int(packet.frame_info.len)
    feature_vector.time_interval = float(packet.frame_info.time_epoch) - last_time
    feature_vector.protocol = int(get_protocol_string(packet.frame_info.protocols))
    feature_vector.dport = get_dport(packet)
    feature_vector.direction = get_direction(packet)

    return feature_vector
