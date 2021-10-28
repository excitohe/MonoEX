from . import transforms as T


def build_transforms(cfg, is_train=True):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD,
                to_bgr=cfg.INPUT.TO_BGR,
            ),
        ]
    )
    return transform
