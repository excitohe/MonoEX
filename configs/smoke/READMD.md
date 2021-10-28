# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation (CVPR2020)


## Citation

```
@InProceedings{SMOKE,
    author    = {Liu, Zechen and Wu, Zizhang and Toth, Roland},
    title     = {SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2020}
}
```


## Train & Evaluate

Single GPU training:
```
CUDA_VISIBLE_DEVICES=7 python tools/train.py --batch_size 8 --config configs/smoke/smoke_dla34_gn.yml --output output/smoke_dla34_gn
```

The model will be evaluated periodically (can be adjusted in $CONFIG) during training and you can also evaluate a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/smoke_dla34_gn.yml --ckpt YOUR_CKPT --eval
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
        <td div align="center">18.40%</td> 
        <td div align="center">14.17%</td> 
        <td div align="center">11.85%</td> 
        <td div align="center">7.22%</td> 
        <td div align="center">5.25%</td> 
        <td div align="center">4.24%</td> 
        <td div align="center">4.30%</td> 
        <td div align="center">2.48%</td> 
        <td div align="center">2.15%</td>  
    </tr>
    <tr>
        <td div align="center">BEV</td>
        <td div align="center">26.91%</td> 
        <td div align="center">21.03%</td> 
        <td div align="center">18.08%</td> 
        <td div align="center">8.39%</td> 
        <td div align="center">6.56%</td> 
        <td div align="center">5.37%</td> 
        <td div align="center">5.03%</td> 
        <td div align="center">2.69%</td> 
        <td div align="center">2.60%</td>  
    </tr>
</table>