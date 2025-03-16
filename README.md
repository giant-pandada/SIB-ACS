# <p align="center">Sampling Innovation-Based Adaptive Compressive Sensing(CVPR 2025)</p>

*<p align="center">Zhifu Tian, Tao Hu, Chaoyang Niu, Di Wu and Shu Wang</p>*

*<p align="center">Information Engineering University, Zhengzhou, China</p>*

*<p align="center">tzhifu@qq.com, hutaoengineering@163.com, niucy2017@outlook.com, wudipaper@sina.com, shu1008@mail.ustc.edu.cn</p>*

## Abstract
Scene-aware Adaptive Compressive Sensing (ACS) has attracted significant interest due to its promising capability for efficient and high-fidelity acquisition of scene images. ACS typically prescribes adaptive sampling allocation (ASA) based on previous samples in the absence of ground truth. However, when confronting unknown scenes, existing ACS methods often lack accurate judgment and robust feedback mechanisms for ASA, thus limiting the high-fidelity sensing of the scene. In this paper, we introduce a Sampling Innovation-Based ACS (SIB-ACS) method that can effectively identify and allocate sampling to challenging image reconstruction areas, culminating in high-fidelity image reconstruction. An innovation criterion is proposed to judge ASA by predicting the decrease in image reconstruction error attributable to sampling increments, thereby directing more samples towards regions where the reconstruction error diminishes significantly. A sampling innovation-guided multi-stage adaptive sampling (AS) framework is proposed, which iteratively refines the ASA through a multi-stage feedback process. For image reconstruction, we propose a Principal Component Compressed Domain Network (PCCD-Net), which efficiently and faithfully reconstructs images under AS scenarios. Extensive experiments demonstrate that the proposed SIB-ACS method significantly outperforms the state-of-the-art methods in terms of image reconstruction fidelity and visual effects.

## Overview
![Sampling](https://github.com/giant-pandada/SIB-ACS/blob/main/figures/Sampling.png) 

![Reconstruction](https://github.com/giant-pandada/SIB-ACS/blob/main/figures/Reconstruction.png) 

## Environment
```
- python == 3.8.13
- pytorch == 1.11.0
- numpy == 1.22.3
- torchvision == 0.12.0
- timm == 0.6.13
- scikit-image == 0.19.2
- opencv-python == 4.5.5.64
- tqdm == 4.64.1
```

## Test
Preparation 1: 
Set the sampling rate and dataset for the test in the eval.py.

Preparation 2: 
Place the test dataset into the `./dataset/test/` folder and Place the pretrained models into the `./results/10/models/` folder.

Operation: 
Run eval.py.

## Results
![result1](https://github.com/giant-pandada/SIB-ACS/blob/main/figures/result1.png) 

![result2](https://github.com/giant-pandada/SIB-ACS/blob/main/figures/result2.png) 

## Training
Preparation 1: 
Set the sampling rate and batchsize for the training in the train.py.

Preparation 2: 
Place the training data file `train.pt` into the `./dataset` folder.

Operation: 
Firstly, start training from the first lightweight model, setting the phase parameter to 1, and then run `train.py` to train the first-stage model. At the end of each stage of training, the trained model parameter file for the current stage, `net_params_{epoch}`, are obtained. Select the optimal trained model parameter file , rename it to `model.pth`, and delete the parameters from other training epoch. Finally, run `train.py` again to train the next stage model. This process continues until the 9th stage is reached, at which point the final, complete model parameters are obtained.

## Pretrained Models and training data file
Pre-trained models and training data file can be obtained from Baidu disk.

## Citation
```
@Inproceedings{tian,
  author    = {Zhifu Tian and Tao Hu and Chaoyang Niu and Di Wu and Shu Wang},
  title     = {Sampling Innovation-Based Adaptive Compressive Sensing},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```
