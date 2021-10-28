import os


class DatasetCatalog():
    DATA_DIR = "/data2/he_guan/data/"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/training/"
        },
        "kitti_val": {
            "root": "kitti/training/"
        },
        "kitti_test": {
            "root": "kitti/testing/"
        },
    }

    @staticmethod
    def get(name, style):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(root=os.path.join(data_dir, attrs["root"]), )
            if style == "MonoFlex":
                factory = "KITTIMonoFlexDataset"
            elif style == "SMOKE":
                factory = "KITTISMOKEDataset"
            elif style == "GUP":
                factory = "KITTIGUPDataset"
            else:
                raise ValueError(f"Unsupport dataset style: {style}")

            return dict(
                factory=factory,
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {"DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"}

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
