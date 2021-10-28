# MonoEX
MonoEX: Excito Private Monocular 3D Object Detection Lib.

**Work in progress.**

## Installation

```bash
python setup.py develop
```

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──ImageSets
│──training
│    ├──calib
│    ├──label_2 
│    └──image_2
└──testing
     ├──calib
     └──image_2
```  

Then modify the paths in config/paths_catalog.py according to your data path.


## Benchmark

Supported methods and backbones are shown in the below table.

Support backbones:

- [x] DLA

Support methods

- [x] [MonoFlex (CVPR'2021)](configs/monoflex/README.md)
- [x] [SMOKE (CVPR'2020)](configs/smoke/README.md)


## Acknowledgement

The codebase is heavily borrowed from following projects:

- [MonoFlex](https://github.com/zhangyp15/MonoFlex)
- [SMOKE](https://github.com/lzccccc/SMOKE)
- [mmcv](https://github.com/open-mmlab/mmcv)

Thanks for their contribution.