
class Fingerprint:

    def __init__(self):
        self.dport = None
        self.protocol = None
        self.direction = None
        self.frame_len = None
        self.time_interval = None
    

    def is_none(self):
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
