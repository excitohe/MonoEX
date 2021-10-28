import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import leastsq

TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 100
TOP_Z_MIN = -3.5
TOP_Z_MAX = 0.6
TOP_X_DIVISION = 0.05
TOP_Y_DIVISION = 0.05
TOP_Z_DIVISION = 0.02
CBOX = [[0.0, 70.4], [-40.0, 40.0], [-3.0, 2.0]]


def convert_alpha_to_rty(alpha, z3d, x3d):
    if isinstance(z3d, torch.Tensor):
        rty = alpha + torch.atan2(-z3d, x3d) + 0.5 * math.pi
        while torch.any(rty > math.pi):
            rty[rty > math.pi] -= math.pi * 2
        while torch.any(rty <= (-math.pi)):
            rty[rty <= (-math.pi)] += math.pi * 2
    elif isinstance(z3d, np.ndarray):
        rty = alpha + np.arctan2(-z3d, x3d) + 0.5 * math.pi
        while np.any(rty > math.pi):
            rty[rty > math.pi] -= math.pi * 2
        while np.any(rty <= (-math.pi)):
            rty[rty <= (-math.pi)] += math.pi * 2
    else:
        rty = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
        while rty > math.pi:
            rty -= math.pi * 2
        while rty <= (-math.pi):
            rty += math.pi * 2
    return rty


def convert_rty_to_alpha(rty, z3d, x3d):
    if isinstance(z3d, torch.Tensor):
        alpha = rty - torch.atan2(-z3d, x3d) - 0.5 * math.pi
        while torch.any(alpha > math.pi):
            alpha[alpha > math.pi] -= math.pi * 2
        while torch.any(alpha <= (-math.pi)):
            alpha[alpha <= (-math.pi)] += math.pi * 2
    elif isinstance(z3d, np.ndarray):
        alpha = rty - np.arctan2(-z3d, x3d) - 0.5 * math.pi
        while np.any(alpha > math.pi):
            alpha[alpha > math.pi] -= math.pi * 2
        while np.any(alpha <= (-math.pi)):
            alpha[alpha <= (-math.pi)] += math.pi * 2
    else:
        alpha = rty - math.atan2(-z3d, x3d) - 0.5 * math.pi
        while alpha > math.pi:
            alpha -= math.pi * 2
        while alpha <= (-math.pi):
            alpha += math.pi * 2
    return alpha


def refresh_attributes(calib):
    # refresh all attributes when P2 varies
    calib.c_u = calib.P[0, 2]
    calib.c_v = calib.P[1, 2]
    calib.f_u = calib.P[0, 0]
    calib.f_v = calib.P[1, 1]
    calib.b_x = calib.P[0, 3] / (-calib.f_u)
    calib.b_y = calib.P[1, 3] / (-calib.f_v)


class Object3D:

    def __init__(self, line):
        data = line.split(" ")
        data[1:] = [float(x) for x in data[1:]]
        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = data[2]
        self.real_alpha = data[3]
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax], dtype=np.float32)
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = np.array([data[11], data[12], data[13]], dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.t)
        self.rty = data[14]  # yaw (around Y-axis in cam_coord) [-pi..pi]
        self.ray = math.atan2(self.t[0], self.t[2])
        self.alpha = convert_rty_to_alpha(self.rty, self.t[2], self.t[0])
        self.level_str = None  # difficulty level
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1
        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corner_3d(self):
        """
        :param: corner_3d: [8, 3] corners of box3d in cam_coord.
        """
        l, h, w = self.l, self.h, self.w
        corner_x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        corner_y = [0, 0, 0, 0, -h, -h, -h, -h]
        corner_z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        R = np.array([[np.cos(self.rty), 0, np.sin(self.rty)], [0, 1, 0], [-np.sin(self.rty), 0, np.cos(self.rty)]])
        corner_3d = np.vstack([corner_x, corner_y, corner_z])  # [3, 8]
        corner_3d = np.dot(R, corner_3d).T
        corner_3d = corner_3d + self.t
        return corner_3d

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                        self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.t[0], self.t[1], self.t[2],
                        self.rty)
        return kitti_str

    def __repr__(self):
        repr_str = self.__class__.__name__ + ', '
        repr_str += f'type={self.type}, '
        repr_str += f'truncation={self.truncation}, '
        repr_str += f'occlusion={self.occlusion}, '
        repr_str += f'alpha={self.alpha}, '
        repr_str += f'box2d:[xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax}], '
        repr_str += f'dim3d:[h={self.h}, w={self.w}, l={self.l}], '
        repr_str += f'loc3d:[x={self.t[0]}, y={self.t[1]}, z={self.t[2]}], '
        repr_str += f'rty3d={self.rty}, '
        repr_str += f'level={self.level}'
        return repr_str


class Calibration(object):
    """
    Calibration utils.
    3D_XYZ in <label>.txt in rect camera coord.
    2D_BOX in <label>.txt in image_2 coord.
    3D_PTS in <lidar>.bin in velodyne coord.
    Formulation:
        y_image2 = P2_rect * x_rect
        y_image2 = P2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P2_rect = [fu, 0,  cu, -fu*bx
                   0,  fv, cv, -fv*by
                   0,  0,  1,  0] = K * [1|t]
    Coordination:
        image coord: x-axis (u) y-axis (v)
        velo coord: front x, left y, up z
        rect/ref camera coord: right x, down y, front z
    """

    def __init__(self, calib_file, from_video=False, use_right_cam=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_file)
        else:
            calibs = self.read_calib_file(calib_file)

        # Coordinate projection matrix from RECT_CAMERA to IMAGE
        self.P = calibs["P3"] if use_right_cam else calibs["P2"]
        self.P = np.reshape(self.P, [3, 4])

        # Coordinate rigid transform from VELODYNE to REFERENCE CAMERA
        self.V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = self.inverse_rigid_trans(self.V2C)

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, file_path):
        data = {}
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def read_calib_from_video(self, calib_root_dir):
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, "calib_cam_to_cam.txt"))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, "calib_velo_to_cam.txt"))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R0_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        return data

    def cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart_to_hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart_to_hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def project_rect_to_image(self, pts_3d_rect):
        if isinstance(pts_3d_rect, torch.Tensor):
            pts_3d_rect = torch.cat((pts_3d_rect, torch.ones(pts_3d_rect.shape[0], 1).type_as(pts_3d_rect)), dim=1)
            pts_2d = torch.matmul(pts_3d_rect, torch.from_numpy(self.P).type_as(pts_3d_rect).t())  # nx3
        else:
            pts_3d_rect = self.cart_to_hom(pts_3d_rect)
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2], pts_2d[:, 2]

    def project_velo_to_image(self, pts_3d_velo):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        y0 = max(0, y0)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    def project_image_to_rect(self, uvd):
        """
        Args:
            uvd: [N, 3] 1st/2nd channel are u/v, 3rd channel 
                is depth in rect camera coord.
        Return:
            pts_3d_rect: [N, 3], points in rect camera coord.
        """
        n = uvd.shape[0]
        x = ((uvd[:, 0] - self.c_u) * uvd[:, 2]) / self.f_u + self.b_x
        y = ((uvd[:, 1] - self.c_v) * uvd[:, 2]) / self.f_v + self.b_y
        if isinstance(uvd, np.ndarray):
            pts_3d_rect = np.zeros((n, 3))
        else:
            pts_3d_rect = uvd.new(uvd.shape).zero_()
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uvd[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uvd):
        pts_3d_rect = self.project_image_to_rect(uvd)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        depth_pt3d = self.get_depth_pt3d(depth)
        depth_UVD = np.zeros_like(depth_pt3d)
        depth_UVD[:, 0] = depth_pt3d[:, 1]
        depth_UVD[:, 1] = depth_pt3d[:, 0]
        depth_UVD[:, 2] = depth_pt3d[:, 2]
        depth_velo = self.project_image_to_velo(depth_UVD)
        if constraint_box:
            depth_box_fov_inds = (
                (depth_velo[:, 0] < CBOX[0][1]) & (depth_velo[:, 0] >= CBOX[0][0])
                & (depth_velo[:, 1] < CBOX[1][1]) & (depth_velo[:, 1] >= CBOX[1][0]) & (depth_velo[:, 2] < CBOX[2][1])
                & (depth_velo[:, 2] >= CBOX[2][0])
            )
            depth_velo = depth_velo[depth_box_fov_inds]
        return depth_velo

    @staticmethod
    def inverse_rigid_trans(Tr):
        """
        Inverse a rigid body transform matrix (3x4 as [R|t]) [R'|-R't; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    @staticmethod
    def get_depth_pt3d(depth):
        pt3d = []
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                pt3d.append([i, j, depth[i, j]])
        return np.array(pt3d)

    # Deprecation: GUP USE
    def alpha2ry(self, alpha, u):
        ry = alpha + np.arctan2(u - self.c_u, self.f_u)
        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi
        return ry

    # Deprecation: GUP USE
    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.c_u, self.f_u)
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi
        return alpha

    # Deprecation: GUP USE
    def flip(self, image_size):
        w_size = 4
        h_size = 2
        p2ds = np.concatenate(
            [
                np.expand_dims(np.tile(np.expand_dims(np.linspace(0, image_size[0], w_size), 0), [h_size, 1]), -1),
                np.expand_dims(np.tile(np.expand_dims(np.linspace(0, image_size[1], h_size), 1), [1, w_size]), -1),
                np.linspace(2, 78, w_size * h_size).reshape(h_size, w_size, 1)
            ], -1
        ).reshape(-1, 3)
        p3ds = self.project_image_to_rect(p2ds)
        p3ds[:, 0] *= -1
        p2ds[:, 0] = image_size[0] - p2ds[:, 0]

        cos_mat = np.zeros([w_size * h_size, 2, 7])
        cos_mat[:, 0, 0] = p3ds[:, 0]
        cos_mat[:, 0, 1] = cos_mat[:, 1, 2] = p3ds[:, 2]
        cos_mat[:, 1, 0] = p3ds[:, 1]
        cos_mat[:, 0, 3] = cos_mat[:, 1, 4] = 1
        cos_mat[:, :, -2] = -p2ds[:, :2]
        cos_mat[:, :, -1] = (-p2ds[:, :2] * p3ds[:, 2:3])
        new_calib = np.linalg.svd(cos_mat.reshape(-1, 7))[-1][-1]
        new_calib /= new_calib[-1]

        new_calib_matrix = np.zeros([4, 3]).astype(np.float32)
        new_calib_matrix[0, 0] = new_calib_matrix[1, 1] = new_calib[0]
        new_calib_matrix[2, 0:2] = new_calib[1:3]
        new_calib_matrix[3, :] = new_calib[3:6]
        new_calib_matrix[-1, -1] = self.P2[-1, -1]

        self.P = new_calib_matrix.T
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """
    Transforation matrix from rotation matrix and translation vector.
    """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def angle_to_class(angle, num_head_bin=12):
    """
    Convert continuous angle to discrete class and residual.
    """
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_head_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class_to_angle(cls, residual, label_format=False, num_head_bin=12):
    """
    Inverse function angle_to_class
    """
    angle_per_class = 2 * np.pi / float(num_head_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3D(line) for line in lines]
    return objects


def load_velo_scan(velo_filename, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=np.float32).reshape((-1, n_vec))
    return scan


def lidar_to_top_coords(x, y, z=None):
    if 0:
        return x, y
    else:
        X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)
        return xx, yy


def lidar_to_top(lidar):
    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)
    if 1:  # new method
        for x in range(Xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x
            for y in range(Yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y
                top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                top[yy, xx, Zn] = quantized_xy[max_height_point, 3]
                for z in range(Zn):
                    iz = np.where((quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1))
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z
                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    top[yy, xx, zz] = max_height
    return top


def project_3d_to_2d(pts):
    x0 = np.min(pts[:, 0])
    x1 = np.max(pts[:, 0])
    y0 = np.min(pts[:, 1])
    y1 = np.max(pts[:, 1])
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def project_to_image(pts_3d, P):
    """ 
    Project 3d points to image plane.
    Usage:
        pts_2d = projectToImage(pts_3d, P)
        :param: pts_3d: nx3 matrix
        :param: P: 3x4 projection matrix
        :return: pts_2d: nx2 matrix
    P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
        => normalize projected_pts_2d(2xn)
    <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
        => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box3d(obj, P):
    """ 
    Takes an object and a projection matrix (P) and projects box3d 
    into the image plane.
    Returns:
        corners_2d: (8,2) array in left image coord.
        corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.rty)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orient3d(obj, P):
    """ 
    Takes an object and a projection matrix (P) and projects the 3d
    object orientation vector into the image plane.
    Returns:
        orientation_2d: (2,2) array in left image coord.
        orientation_3d: (2,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = roty(obj.rty)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=4):
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def draw_projected_box3d(
    image,
    qs,
    color=None,
    cls=None,
    thickness=1,
    draw_orientation=False,
    draw_height_text=False,
    draw_center_line=False,
    draw_corner=True,
    draw_number=False,
    corner_size=3,
    draw_corner_line=False,
    draw_orien_surface=True
):
    """ 
    Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """

    if cls is not None and color is None:
        if cls == 'Car':
            color = (0, 255, 255)
        elif cls == 'Pedestrian':
            color = (255, 225, 255)
        elif cls == 'Cyclist':
            color = (139, 0, 0)

    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    # draw the front: x = l / 2, link 0 <-> 4 and 1 <-> 5
    if draw_orientation:
        cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1]), color, thickness, cv2.LINE_AA)
        cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1]), color, thickness, cv2.LINE_AA)

    if draw_orien_surface:
        orien_mask = np.zeros((image.shape), dtype=np.uint8)
        # orien contours
        contours = qs[[0, 1, 5, 4], :].astype(np.int64)
        orien_mask = cv2.fillPoly(orien_mask, pts=[contours], color=color)

        image = cv2.addWeighted(image, 1, orien_mask, 0.3, 0)

    # draw colorful vertices, 0=4, 1=5, 2=6, 3=7, 9=10
    colors = [
        (43, 253, 63), (227, 34, 243), (254, 117, 34), (30, 255, 254), (43, 253, 63), (227, 34, 243), (254, 117, 34),
        (30, 255, 254), (255, 216, 49), (255, 216, 49)
    ]

    if draw_center_line:
        draw_dotted_line(image, (qs[9, 0], qs[9, 1]), (qs[8, 0], qs[8, 1]), colors[-1], 1, cv2.LINE_AA)

    if draw_height_text:
        height_text_size = 0.5
        height_pos = (qs[:4] + qs[4:8]) / 2
        height_pos_center = qs[8:].mean(axis=0)
        for i in range(height_pos.shape[0]):
            cv2.putText(
                image, 'h_{}'.format(i + 1), tuple(height_pos[i].astype(np.int)), cv2.FONT_HERSHEY_SIMPLEX,
                height_text_size, (255, 0, 0), 1, cv2.LINE_AA
            )

        cv2.putText(
            image, 'h_c'.format(i + 1), tuple(height_pos_center.astype(np.int)), cv2.FONT_HERSHEY_SIMPLEX,
            height_text_size, (255, 0, 0), 1, cv2.LINE_AA
        )

    if type(draw_corner) == bool:
        if draw_corner:
            for i in range(qs.shape[0]):
                cv2.circle(image, tuple(qs[i]), corner_size, colors[i], -1)
                if draw_corner_line:
                    if i in [0, 1, 2, 3]:
                        cv2.line(
                            image, (qs[i, 0], qs[i, 1]), (qs[i + 4, 0], qs[i + 4, 1]), colors[i], thickness, cv2.LINE_AA
                        )
                    elif i == 8:
                        cv2.line(
                            image, (qs[i, 0], qs[i, 1]), (qs[i + 1, 0], qs[i + 1, 1]), colors[i], thickness, cv2.LINE_AA
                        )
                if draw_number:
                    cv2.putText(image, str(i), tuple(qs[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    elif type(draw_corner) == list:
        for i in draw_corner:
            cv2.circle(image, tuple(qs[i]), corner_size, colors[i], -1)
            if draw_corner_line:
                if i in [0, 1, 2, 3]:
                    cv2.line(
                        image, (qs[i, 0], qs[i, 1]), (qs[i + 4, 0], qs[i + 4, 1]), colors[i], thickness, cv2.LINE_AA
                    )
                elif i == 8:
                    cv2.line(
                        image, (qs[i, 0], qs[i, 1]), (qs[i + 1, 0], qs[i + 1, 1]), colors[i], thickness, cv2.LINE_AA
                    )
            if draw_number:
                cv2.putText(image, str(i), tuple(qs[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    return image


def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    divisor = np.max(top_image) - np.min(top_image)
    top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image


def init_bev_image(out_size=768):
    bird_view = np.ones((out_size, out_size, 3), dtype=np.uint8) * 230
    return bird_view


def draw_bev_box3d(
    image,
    boxes3d,
    color=(255, 255, 255),
    cls=None,
    thickness=1,
    scores=None,
    text_lables=[],
    world_size=64,
    out_size=768
):
    if color is None:
        if cls is not None:
            if cls == 'Car':
                color = (0, 255, 255)
            elif cls == 'Pedestrian':
                color = (255, 225, 255)
            elif cls == 'Cyclist':
                color = (139, 0, 0)
        else:
            color = (0, 0, 255)
    for idx in range(boxes3d.shape[0]):
        pts = boxes3d[idx][[0, 1, 2, 3]][:, [0, 2]]
        pts[:, 0] += world_size / 2
        pts[:, 1] = world_size - pts[:, 1]
        pts = (pts * out_size / world_size).astype(np.int32)
        cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, color, 2, lineType=cv2.LINE_AA)
        cv2.line(image, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), color, 4, lineType=cv2.LINE_AA)
    return image


def draw_box3d_on_top(image, boxes3d, color=(255, 255, 255), thickness=1, scores=None, text_lables=[], is_gt=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
    return img


def hypothesis_func(w, x):
    w1, w0 = w
    return w1 * x + w0


def error_func(w, train_x, train_y):
    return hypothesis_func(w, train_x) - train_y


def dump_fit_func(w_fit):
    w1, w0 = w_fit
    print("fitting line=", str(w1) + "*x + " + str(w0))
    return


def linear_regression(train_x, train_y, test_x):
    w_init = [20, 1]  # weight factor init
    fit_ret = leastsq(error_func, w_init, args=(train_x, train_y))
    w_fit = fit_ret[0]
    # dump fit result
    dump_fit_func(w_fit)
    test_y = hypothesis_func(w_fit, test_x)
    test_y0 = hypothesis_func(w_fit, train_x)
    return test_y, test_y0


def get_iou3d(k_corner_3d, q_corner_3d, need_bev=False):
    """	
    :param k_corner_3d: (N, 8, 3) in rect coords	
    :param q_corner_3d: (M, 8, 3)
    """
    from shapely.geometry import Polygon
    A, B = k_corner_3d, q_corner_3d
    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype=np.float32)
    iou_bev = np.zeros((N, M), dtype=np.float32)
    # for height overlap, since y face down, use the negative y
    min_h_a = -A[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_a = -A[:, 4:8, 1].sum(axis=1) / 4.0
    min_h_b = -B[:, 0:4, 1].sum(axis=1) / 4.0
    max_h_b = -B[:, 4:8, 1].sum(axis=1) / 4.0
    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_h_a[i], min_h_b[j]])
            min_of_max = np.min([max_h_a[i], max_h_b[j]])
            h_overlap = np.max([0, min_of_max - max_of_min])
            if h_overlap == 0:
                continue
            bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]].T), Polygon(B[j, 0:4, [0, 2]].T)
            if bottom_a.is_valid and bottom_b.is_valid:
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.0
            overlap3d = bottom_overlap * h_overlap
            union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + \
                      bottom_b.area * (max_h_b[j] - min_h_b[j]) - overlap3d
            iou3d[i][j] = overlap3d / union3d
            iou_bev[i][j] = bottom_overlap / (bottom_a.area + bottom_b.area - bottom_overlap)
    if need_bev:
        return iou3d, iou_bev
    return iou3d


def draw_trunc_heatmap(project_center, box2d, image_size):
    center_2d = (box2d[:2] + box2d[2:]) / 2


def approx_project_center(project_center, surface_center, image_size):
    # surface_inside
    image_w, image_h = image_size
    surface_center_inside_image = (
        (surface_center[:, 0] >= 0) & (surface_center[:, 1] >= 0) & (surface_center[:, 0] <= image_w - 1) &
        (surface_center[:, 1] <= image_h - 1)
    )
    if surface_center_inside_image.sum() > 0:
        target_surface_center = surface_center[surface_center_inside_image.argmax()]
        # y = ax + b
        a, b = np.polyfit(
            [project_center[0], target_surface_center[0]], [project_center[1], target_surface_center[1]], 1
        )
        valid_intersects = []
        valid_edge = []
        left_y = b
        if (0 <= left_y <= image_h - 1):
            valid_intersects.append(np.array([0, left_y]))
            valid_edge.append(0)
        right_y = (image_w - 1) * a + b
        if (0 <= right_y <= image_h - 1):
            valid_intersects.append(np.array([image_w - 1, right_y]))
            valid_edge.append(1)
        top_x = -b / a
        if (0 <= top_x <= image_w - 1):
            valid_intersects.append(np.array([top_x, 0]))
            valid_edge.append(2)
        bottom_x = (image_h - 1 - b) / a
        if (0 <= bottom_x <= image_w - 1):
            valid_intersects.append(np.array([bottom_x, image_h - 1]))
            valid_edge.append(3)
        valid_intersects = np.stack(valid_intersects)
        min_idx = np.argmin(np.linalg.norm(valid_intersects - project_center.reshape(1, 2), axis=1))
        return valid_intersects[min_idx], valid_edge[min_idx]
    else:
        return None


def get_3d_dis(image, objs, calib):
    image = np.array(image)
    obj = objs[0]
    point_radius = 2

    corners_3d = obj.generate_corners3d()
    corners_2d, _ = calib.project_rect_to_image(corners_3d)

    box2d = project_3d_to_2d(corners_2d).astype(np.int)

    center_3d = obj.t.copy()
    center_3d[1] = center_3d[1] - obj.h / 2
    center_2d, _ = calib.project_rect_to_image(center_3d.reshape(-1, 3))
    center_2d = center_2d.astype(np.int)

    center_idxs = np.array([[0, 1, 4, 5], [0, 1, 2, 3], [0, 3, 4, 7]])
    # shape: 3 x 4 x 3
    centers = corners_3d[center_idxs].mean(axis=1)
    centers_2d = corners_2d[center_idxs].mean(axis=1).astype(np.int)
    # vectors
    vectors = centers_2d - center_2d.reshape(1, 2)
    vectors = vectors / np.sqrt(np.sum(vectors**2, axis=1)).reshape(-1, 1)

    image = draw_projected_box3d(image, corners_2d, color=(0, 255, 0))
    cv2.circle(image, tuple(center_2d[0].tolist()), point_radius, (255, 0, 0), 0)

    point_list = [tuple(centers_2d[i].tolist()) for i in range(3)]
    for point in point_list:
        cv2.circle(image, point, point_radius, (255, 0, 255), 0)

    line_color = (255, 255, 0)
    thickness = 1
    lineType = 4

    i = np.random.randint(box2d[0], box2d[2])
    j = np.random.randint(box2d[1], box2d[3])

    image_ij = image.copy()
    ptStart = (i, j)
    point = np.array([i, j])
    # center -> point
    local_point = point - center_2d
    cv2.arrowedLine(image_ij, (center_2d[0, 0], center_2d[0, 1]), (point[0], point[1]), line_color, thickness, lineType)
    # projection
    proj_local = vectors * np.sum(local_point * vectors, axis=1).reshape(-1, 1)
    proj_local_point = center_2d + proj_local
    cv2.arrowedLine(
        image_ij, (center_2d[0, 0], center_2d[0, 1]), (int(proj_local_point[0, 0]), int(proj_local_point[0, 1])),
        line_color, thickness, lineType
    )
    bias = local_point - proj_local
    surface_proj = centers_2d + bias
    cv2.arrowedLine(
        image_ij, (centers_2d[0, 0], centers_2d[0, 1]), (int(surface_proj[0, 0]), int(surface_proj[0, 1])), line_color,
        thickness, lineType
    )
    dis = surface_proj - point

    plt.figure()
    plt.imshow(image_ij)
    plt.show()


def show_image_with_boxes(
    image,
    cls_ids,
    target_center,
    box2d,
    corners_2d,
    reg_mask,
    offset_3D,
    down_ratio,
    pad_size,
    encoded_alphas,
    vis=True,
    index=[]
):
    image = np.array(image)
    img2 = np.copy(image)  # for 2d bbox
    img3 = np.copy(image)  # for 3d bbox
    font = cv2.FONT_HERSHEY_SIMPLEX

    from umi.utils.visualizer import Visualizer
    img2_vis = Visualizer(img2)

    ori_target_center = (target_center + offset_3D) * down_ratio

    box2d[:, 0::2] = box2d[:, 0::2] * down_ratio
    box2d[:, 1::2] = box2d[:, 1::2] * down_ratio
    center_2d = (box2d[:, :2] + box2d[:, 2:]) / 2

    id_to_cls = ['Car', 'Pedestrian', 'Cyclist']

    for idx in range(target_center.shape[0]):
        if reg_mask[idx] == 0:
            continue

        img2_vis.draw_box(box_coord=box2d[idx])

        alpha_regress = encoded_alphas[idx]
        text = '{} {} {:.1f}'.format(alpha_regress[0], alpha_regress[1], alpha_regress[2] / np.pi * 180)
        # img2_vis.draw_text(text=text, position=(int(box2d[idx, 0]), int(box2d[idx, 1])))

        # 8 x 3
        corners_2d_i = (corners_2d[idx][:, :2] + target_center[idx].reshape(1, 2)) * down_ratio
        img3 = draw_projected_box3d(img3, corners_2d_i, cls=id_to_cls[cls_ids[idx]], draw_corner=index)

        # target center and 3D center
        cv2.circle(img3, tuple(center_2d[idx].astype(np.int)), 4, (255, 0, 0), -1)
        cv2.circle(img3, tuple(ori_target_center[idx].astype(np.int)), 4, (0, 255, 255), -1)

    img2 = img2_vis.output.get_image()
    stacked_img = np.vstack((img2, img3))

    if vis:
        plt.figure(figsize=(10, 6))
        plt.imshow(stacked_img)
        plt.show()

    return img3


def show_edge_heatmap(img, edge_heatmap, interp_edge_indices, output_size):
    # output_size: w, h
    resized_img = img.resize(output_size)
    interp_edge_indices_int = interp_edge_indices.round().astype(np.int)
    full_edgemap = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    full_edgemap[interp_edge_indices_int[:, 1], interp_edge_indices_int[:, 0]] = edge_heatmap[0]

    plt.figure()
    plt.subplot(211)
    plt.imshow(resized_img)
    plt.subplot(212)
    plt.imshow(full_edgemap)
    plt.show()


def show_heatmap(img, heat_map, classes=['Car', 'Pedestrian', 'Cyclist'], index=None):
    colors = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
    ignored_color = np.array([1, 1, 1])

    resized_img = img.resize((heat_map.shape[2], heat_map.shape[1]))
    mix_img = np.array(resized_img) / 255
    all_heat_img = np.zeros_like(mix_img)

    for k in range(len(classes)):
        heat_map_k = heat_map[k, :, :]
        ignored_class_map = np.zeros_like(mix_img)
        ignored_class_map[heat_map_k == -1] = ignored_color
        heat_map_k[heat_map_k == -1] = 0
        class_map = heat_map_k[..., np.newaxis] * np.array(colors[k]).reshape(1, 1, 3) + ignored_class_map
        mix_img += class_map
        all_heat_img += class_map

    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(mix_img)
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(all_heat_img)
    plt.axis('off')
    plt.show()
