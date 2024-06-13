
class FeatureVector:

    def __init__(self):
        """
        Initialise the 5 features extracted from each packet.
        """
        self.dport = None
        self.protocol = []
        self.direction = None
        self.frame_len = None
        self.time_interval = None
    