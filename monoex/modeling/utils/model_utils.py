class MakeKeyValuePair(object):

    def __init__(self, keys, chns):
        super(MakeKeyValuePair, self).__init__()

        self.keys = [key for key_group in keys for key in key_group]
        self.chns = [chn for chn_group in chns for chn in chn_group]

    def __call__(self, key):
        index = self.keys.index(key)
        s = sum(self.chns[:index])
        e = s + self.chns[index]
        return slice(s, e, 1)


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    """
    NOTE: this is a numpy function
    Args:
        dets (array): shape of [batch, max_dets, dim]
        info (dict): necessary info of input images
    """
    results = {}
    pass