from .configs import CfgNode, configurable, get_cfg, global_cfg, set_global_cfg

KITTI_TYPE_ID_CONVERSION = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': -4,
    'Truck': -4,
    'Person_sitting': -2,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1,
}

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "configurable",
    "KITTI_TYPE_ID_CONVERSION",
]
