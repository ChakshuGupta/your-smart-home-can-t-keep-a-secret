import ipaddress
from src.object.fingerprint import Fingerprint
  

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
        elif proto == "tls":
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
        print(packet.layers)
        src = ipaddress.ip_address(packet.ipv6.src)
        dst = ipaddress.ip_address(packet.ipv6.dst)

        if src.is_global:
            return 0
        elif dst.is_global:
            return 1
        else:
            return 0

    elif "IP" in str(packet.layers):
        print(packet.layers)
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
    
    return packet[packet.transport_layer].dstport

def extract_features(packet, last_time):
    """
    Extract the features required from the packet and store it
    in the object of the fingerprint class.
    """
    fingerprint = Fingerprint()

    fingerprint.frame_len = packet.frame_info.len
    fingerprint.time_interval = float(packet.frame_info.time_epoch) - last_time
    fingerprint.protocol = get_protocol_string(packet.frame_info.protocols)
    fingerprint.dport = get_dport(packet)
    fingerprint.direction = get_direction(packet)

    return fingerprint