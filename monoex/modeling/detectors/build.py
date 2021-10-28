from monoex.utils.registry import Registry

DETECTOR_REGISTRY = Registry("DETECTOR")


def build_detector(cfg):
    """
    Build a detector from `cfg.MODEL.DETECTOR_ARCH`.
    """
    detector_name = cfg.MODEL.DETECTOR_ARCH
    detector = DETECTOR_REGISTRY.get(detector_name)(cfg)
    return detector
