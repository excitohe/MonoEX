import torch
import torch.nn as nn
"""
References:
    Learning Depth-guided Convolution for Monocular 3D Object Detection, CVPR'20
"""


def shift_pool(feats, shift_times=3):
    update_feats = feats.clone()
    for i in range(1, shift_times):
        update_feats += torch.cat([feats[:, i:, :, :], feats[:, :i, :, :]], dim=1)
    update_feats /= shift_times
    return update_feats


if __name__ == '__main__':
    f1 = torch.ones(1, 1, 3, 3)
    f2 = f1 + 1
    f3 = f2 + 1
    origin_feats = torch.cat([f1, f2, f3], dim=1)
    print('origin_feats:\n', origin_feats)

    update_feats = shift_pool(origin_feats, 3)
    print('update_feats:\n', update_feats)