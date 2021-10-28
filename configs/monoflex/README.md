# Objects are Different: Flexible Monocular 3D Object Detection (CVPR2021)


## Citation

```
@InProceedings{MonoFlex,
    author    = {Zhang, Yunpeng and Lu, Jiwen and Zhou, Jie},
    title     = {Objects Are Different: Flexible Monocular 3D Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3289-3298}
}
```


## Train & Evaluate

Single GPU training:
```
CUDA_VISIBLE_DEVICES=6 python tools/train.py --batch_size 8 --config configs/monoflex/monoflex_dla34.yml --output output/monoflex_dla34
```

Multi GPU training:
```
CUDA_VISIBLE_DEVICES=0,1 python tools/train.py --num_gpus 2 --batch_size 16 --config configs/monoflex/monoflex_dla34.yml --output output/MonoFlex_DLA34_GPU2
```

The model will be evaluated periodically (can be adjusted in $CONFIG) during training and you can also evaluate a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/monoflex_dla34.yml --ckpt YOUR_CKPT --eval
```

Tips: You can also specify --vis when evaluation to visualize the predicted heatmap and 3d bounding boxes.


## Results

* Environments: single NVIDIA GeForce RTX 3090 GPU.

* The performance on KITTI 3D detection (3D/BEV)(AP@R<sub>40</sub>)on the validation set.

<table align="center">
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Car@IoU=0.7</td>    
        <td colspan="3",div align="center">Pedestrian@IoU=0.5</td>  
        <td colspan="3",div align="center">Cyclist@IoU=0.5</td>  
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Easy</td> 
        <td div align="center">Mod</td> 
        <td div align="center">Hard</td>  
    </tr>
    <tr>
        <td div align="center">3D</td>
        <td div align="center">22.60%</td> 
        <td div align="center">16.10%</td> 
        <td div align="center">13.35%</td> 
        <td div align="center">9.36%</td> 
        <td div align="center">7.19%</td> 
        <td div align="center">5.85%</td> 
        <td div align="center">4.57%</td> 
        <td div align="center">2.34%</td> 
        <td div align="center">2.06%</td>  
    </tr>
    <tr>
        <td div align="center">BEV</td>
        <td div align="center">32.31%</td> 
        <td div align="center">23.50%</td> 
        <td div align="center">20.11%</td> 
        <td div align="center">11.20%</td> 
        <td div align="center">8.86%</td> 
        <td div align="center">7.24%</td> 
        <td div align="center">5.29%</td> 
        <td div align="center">2.70%</td> 
        <td div align="center">2.45%</td>  
    </tr>
</table>