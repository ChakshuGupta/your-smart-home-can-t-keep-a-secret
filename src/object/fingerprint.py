
class Fingerprint:

    def __init__(self):
        """
        Initialise the 5 features extracted from each packet.
        """
        self.dport = None
        self.protocol = None
        self.direction = None
        self.frame_len = None
        self.time_interval = None
    

    def is_none(self):
        """
        Check if any values are none.
        """
        none = False
        if self.dport is None:
            none = True
        if self.protocol is None:
            none = True
        if self.frame_len is None:
            none = True
        if self.direction is None:
            none = True
        if self.time_interval is None:
            none = True

        return none
