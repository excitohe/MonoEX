import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform


def get_transfrom_matrix(center_scale, output_size):
    center, scale = center_scale[0], center_scale[1]
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])
    get_matrix = transform.estimate_transform("affine", src, dst)
    matrix = get_matrix.params
    return matrix.astype(np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(point, mat):
    point = point.reshape(-1, 2)
    point_exd = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)
    point_mat = np.matmul(point_exd, mat.T)
    return point_mat[:, :2].squeeze()


def gaussian_radius(height, width, min_overlap=0.7):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian_round(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # generate meshgrid
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_ellip(shape, sigma_x, sigma_y):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # generate meshgrid
    h = np.exp(-(x * x) / (2 * sigma_x * sigma_x) - (y * y) / (2 * sigma_y * sigma_y))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_1d(edgemap, center, radius):
    diameter = 2 * radius + 1
    sigma = diameter / 6
    gaussian_1d = np.arange(-radius, radius + 1)
    gaussian_1d = np.exp(-(gaussian_1d * gaussian_1d) / (2 * sigma * sigma))
    # 1D mask
    left, right = min(center, radius), min(len(edgemap) - center, radius + 1)
    masked_edgemap = edgemap[center - left:center + right]
    masked_gaussian = gaussian_1d[radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_edgemap.shape) > 0:
        np.maximum(masked_edgemap, masked_gaussian, out=masked_edgemap)
    return edgemap


def draw_umich_gaussian_round(heatmap, center, radius, k=1, ignore=False):
    diameter = 2 * radius + 1
    gaussian = gaussian_round((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        if ignore:
            masked_heatmap[masked_heatmap == 0] = -1
        else:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_umich_gaussian_ellip(heatmap, center, radius_x, radius_y, k=1):
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian = gaussian_ellip((diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def sigmoid_hm(heatmap_feat):
    x = heatmap_feat.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)
    return x


def nms_hm(heatmap, kernel=3, reso=1):
    kernel = int(kernel / reso)
    if kernel % 2 == 0:
        kernel += 1
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=(kernel, kernel), stride=1, padding=pad)
    eq_index = (hmax == heatmap).float()
    return heatmap * eq_index


def select_topk(heatmap, K=100):
    # NOTE: GUP is little different from MonoFlex
    batch, cat, height, width = heatmap.size()
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).float()
    topk_xs = (topk_inds % width).float()
    assert isinstance(topk_xs, torch.cuda.FloatTensor)
    assert isinstance(topk_ys, torch.cuda.FloatTensor)

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clsid = (topk_ind / K).float()
    assert isinstance(topk_clsid, torch.cuda.FloatTensor)

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clsid, topk_ys, topk_xs


def _gather_feat(feat, inds, mask=None):
    """
    Args:
        feat (tensor): shape of [B, H*W, C]
        inds (tensor): shape of [B, K]
        mask (tensor): shape of [B, K] or [B, sum]
    """
    dim = feat.size(-1)
    ind = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, inds):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, inds)
    return feat


def select_point_of_interest(batch, index, feats):
    """
    Args:
        batch (int): batch size
        index (tensor): point format or index format
        feats (tensor): shape of [N, C, H, W]
    """
    w = feats.shape[3]
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = index.view(batch, -1)
    # [N, C, H, W] -----> [N, H, W, C]
    feats = feats.permute(0, 2, 3, 1).contiguous()
    channel = feats.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feats = feats.view(batch, -1, channel)
    # expand index in channels
    index = index.unsqueeze(-1).repeat(1, 1, channel)
    # select specific features bases on POIs
    feats = feats.gather(1, index.long())
    return feats


def get_source_tensor(inputs, inds, masks):
    inputs = _transpose_and_gather_feat(inputs, inds)
    return inputs[masks]


def get_target_tensor(target, mask):
    return target[mask]


if __name__ == '__main__':
    import random
    np.set_printoptions(suppress=True, precision=6)
    size = np.array([1242, 370], dtype=np.float32)
    center = size / 2

    input_h = 384
    input_w = 1280

    shift = 0.2
    scale = 0.4

    shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
    center[0] += size[0] * random.choice(shift_ranges)
    center[1] += size[1] * random.choice(shift_ranges)
    scale_ranges = np.arange(1.0 - scale, 1.0 + scale + 0.1, 0.1)
    size *= random.choice(scale_ranges)

    print('center: ', center)
    print('size: ', size)

    center_size = [center, size]
    trans = get_transfrom_matrix(center_size, [input_w, input_h])
    print('trans:\n', trans)

    box2d = np.array([100, 120, 200, 240], dtype=np.float32)
    box2d_a = affine_transform(box2d[:2], trans)
    print('box2d_a:\n', box2d_a)
