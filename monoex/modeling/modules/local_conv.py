import torch
import torch.nn as nn
import torch.nn.functional as F
"""
References:
    1. Monocular 3D Region Proposal Network for Object Detection, ICCV'19
    2. torch.Tensor.unfold (Python method, in torch.Tensor)
        :: unfold(dimension, size, step) → Tensor
        Returns a view of the original tensor which contains all slices of size
        `size` from `self` tensor in the dimension `dimension`.
        :: Example:
        >>> x = torch.arange(1., 8)
        >>> x
        tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> x.unfold(0, 2, 1)
        tensor([[ 1.,  2.],
                [ 2.,  3.],
                [ 3.,  4.],
                [ 4.,  5.],
                [ 5.,  6.],
                [ 6.,  7.]])
        >>> x.unfold(0, 2, 2)
        tensor([[ 1.,  2.],
                [ 3.,  4.],
                [ 5.,  6.]])
"""


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feat_inp, num_feat_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()
        self.num_rows = num_rows
        self.out_chns = num_feat_out
        self.kernel = kernel
        self.padding = padding
        self.group_conv = nn.Conv2d(num_feat_inp * num_rows, num_feat_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):
        batch, chans, feat_h, feat_w = x.size()
        if self.pad:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='constant', value=0)
        bins = int(feat_h / self.num_rows)

        x = x.unfold(2, bins + self.pad * 2, bins)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(batch, chans * self.num_rows, bins + self.pad * 2, (feat_w + self.pad * 2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(batch, self.num_rows, self.out_chns, bins, feat_w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(batch, self.out_chns, feat_h, feat_w)

        return y


if __name__ == '__main__':
    pass
