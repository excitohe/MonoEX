from monoex.utils.registry import Registry

HEAD_REGISTRY = Registry("HEAD")


def build_head(cfg, in_channels):
    """
    Build a head from `cfg.MODEL.HEAD.NAME`.
    """
    head_name = cfg.MODEL.HEAD.NAME
    head = HEAD_REGISTRY.get(head_name)(cfg, in_channels)
    return head
