import logging
import os
import numpy as np
import torch.utils.data as data
from PIL import Image

from .kitti_utils import Calibration, read_label


class KITTIBaseDataset(data.Dataset):

    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTIBaseDataset, self).__init__()

        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.image_right_dir = os.path.join(root, "image_3")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")

        self.split = cfg.DATASETS.TRAIN_SPLIT if is_train else cfg.DATASETS.TEST_SPLIT
        self.is_train = is_train
        self.transforms = transforms

        self.imageset_txt = os.path.join(root, "../ImageSets", "{}.txt".format(self.split))
        assert os.path.exists(self.imageset_txt), "Not exist dir = {}".format(self.imageset_txt)
        image_files = []
        for line in open(self.imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)

        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.class_names = cfg.DATASETS.CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.num_samples = len(self.image_files)

        self.use_right_image = cfg.DATASETS.USE_RIGHT_IMAGE & is_train

        self.input_w = cfg.INPUT.WIDTH_TRAIN
        self.input_h = cfg.INPUT.HEIGHT_TRAIN
        self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_w = self.input_w // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_h = self.input_h // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_size = [self.output_w, self.output_h]
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Init KITTI {self.split} set with {self.num_samples} files loaded.")

    def __len__(self):
        if self.use_right_image:
            return self.num_samples * 2
        else:
            return self.num_samples

    def get_image(self, index, use_right_cam=False):
        if use_right_cam:
            image_file = os.path.join(self.image_right_dir, self.image_files[index])
        else:
            image_file = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_file).convert('RGB')
        return image

    def get_calib(self, index, use_right_cam=False):
        calib_file = os.path.join(self.calib_dir, self.label_files[index])
        calib = Calibration(calib_file, use_right_cam=use_right_cam)
        return calib

    def get_annos(self, index):
        if self.split != 'test':
            label_file = os.path.join(self.label_dir, self.label_files[index])
        annos = read_label(label_file)
        return annos

    def select_whitelist(self, obj_list):
        type_whitelist = self.class_names
        valid_obj_list = []
        for obj in obj_list:
            if obj.type not in type_whitelist:
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    def pad_image(self, image):
        image = np.array(image)
        h, w, c = image.shape
        ret_img = np.zeros((self.input_h, self.input_w, c))
        pad_h = (self.input_h - h) // 2
        pad_w = (self.input_w - w) // 2
        ret_img[pad_h:pad_h + h, pad_w:pad_w + w] = image
        pad_size = np.array([pad_w, pad_h])
        return Image.fromarray(ret_img.astype(np.uint8)), pad_size

    def __getitem__(self, index):
        raise NotImplementedError
