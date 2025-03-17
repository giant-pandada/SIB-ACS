# <p align="center">Sampling Innovation-Based Adaptive Compressive Sensing(CVPR 2025)</p>
*<p align="center">![downloads](https://img.shields.io/github/downloads/{username}/{repo-name}/total.svg)
![stars](https://img.shields.io/github/stars/{username}/{repo-name}.svg)</p>*
*<p align="center">Zhifu Tian, Tao Hu, Chaoyang Niu, Di Wu and Shu Wang</p>*

*<p align="center">Information Engineering University, Zhengzhou, China</p>*

*<p align="center">tzhifu@qq.com, hutaoengineering@163.com, niucy2017@outlook.com, wudipaper@sina.com, shu1008@mail.ustc.edu.cn</p>*

## :dart: Abstract
Scene-aware Adaptive Compressive Sensing (ACS) has attracted significant interest due to its promising capability for efficient and high-fidelity acquisition of scene images. ACS typically prescribes adaptive sampling allocation (ASA) based on previous samples in the absence of ground truth. However, when confronting unknown scenes, existing ACS methods often lack accurate judgment and robust feedback mechanisms for ASA, thus limiting the high-fidelity sensing of the scene. In this paper, we introduce a Sampling Innovation-Based ACS (SIB-ACS) method that can effectively identify and allocate sampling to challenging image reconstruction areas, culminating in high-fidelity image reconstruction. An innovation criterion is proposed to judge ASA by predicting the decrease in image reconstruction error attributable to sampling increments, thereby directing more samples towards regions where the reconstruction error diminishes significantly. A sampling innovation-guided multi-stage adaptive sampling (AS) framework is proposed, which iteratively refines the ASA through a multi-stage feedback process. For image reconstruction, we propose a Principal Component Compressed Domain Network (PCCD-Net), which efficiently and faithfully reconstructs images under AS scenarios. Extensive experiments demonstrate that the proposed SIB-ACS method significantly outperforms the state-of-the-art methods in terms of image reconstruction fidelity and visual effects.

## :loudspeaker: Main Contributions

- A SIB-ACS framework that utilizes incremental image information from historical sampling and sampling increments to determine ASA, thereby enhancing the accuracy of ASA and achieving high-fidelity image reconstruction.

- A SI-guided multi-stage ASA model, which accomplishes multi-stage ASA at any sampling rate with a single model, facilitated by an image domain negative feedback mechanism.

- A PCCD-Net for image reconstruction in ACS scenarios, which leverages PGD operations in the PC image to compress the dimensions of PGD in the FD features, thereby significantly reducing the computational cost while maintaining high image reconstruction performance.

## :gift_heart: Citation
If our work is helpful to you, please consider staring this :star:repository:star: and citing:rose::
```
@Inproceedings{tzfSIBACS,
  author    = {Zhifu Tian and Tao Hu and Chaoyang Niu and Di Wu and Shu Wang},
  title     = {Sampling Innovation-Based Adaptive Compressive Sensing},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## :helicopter: Overview
The adaptive sampling process:

![Sampling](https://github.com/giant-pandada/SIB-ACS_CVPR2025/blob/main/figures/Sampling.png) 

The overview of the proposed sampling innovation-based ASM. (a) Innovation-guided multi-stage AS, (b) Innovation Estimation (IE) based on the reconstructed image information from sampling values before and after Innovation Sampling (IS).

The reconstruction process:

![Reconstruction](https://github.com/giant-pandada/SIB-ACS_CVPR2025/blob/main/figures/Reconstruction.png) 

The overview of the proposed PCCD-Net for image reconstruction. (a) Deep reconstruction process, (b) PCPGD path, (c) CDPGD path, (d) Convolutional block that transitions features from the FD to the CD, (e) Convolutional block that transitions features from the CD back to the FD, (f) Proximal Mapping Module (PMM).

## :egg: Environment
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

## :hatching_chick: Training
:one: Preparation 1: 
Set the parameters for the training in the train.py.

:two: Preparation 2: 
Place the training data file `train.pt` into the `./dataset` folder.

:triangular_flag_on_post: Operation: 
Firstly, start training from the first lightweight model, setting the phase parameter to 1, and then run `train.py` to train the first-stage model. At the end of each stage of training, the trained model parameter file for the current stage, `net_params_{epoch}`, are obtained. Select the optimal trained model parameter file , rename it to `model.pth`, and delete the parameters from other training epoch. Finally, run `train.py` again to train the next stage model. This process continues until the 9th stage is reached, at which point the final, complete model parameters are obtained.

## :baby_chick: Evaluation
:one: Preparation 1: 
Set the sampling rate(any sampling rate between [0.1,0.5]) and dataset for the test in the eval.py.

:two: Preparation 2: 
Place the test dataset into the `./dataset/test/` folder and Place the pretrained models into the `./results/10/models/` folder.

:triangular_flag_on_post: Operation: 
Run eval.py.

## :link: Pretrained Models and Training Data File

- [Pretrained Models](https://pan.baidu.com/s/1RTfLRxqy-embWdtUf6TG7g?pwd=wxkq):`./results/10/models/model.pth`.

- [Training and test datasets](https://pan.baidu.com/s/17XfBHsJJOLR3SKurVRgGVg?pwd=r564):`./datasets/train.pth`, `./datasets/test/BSD68`, `./datasets/test/Urban100`.

## :poultry_leg: Results
The overall performance:

![performance](https://github.com/giant-pandada/SIB-ACS_CVPR2025/blob/main/figures/Performance.png) 

The Visual result of reconstructed images:

![Reconstructed images](https://github.com/giant-pandada/SIB-ACS_CVPR2025/blob/main/figures/Reconstructedimages.png) 

The Visual result of sampling distribution:

![Sampling distribution](https://github.com/giant-pandada/SIB-ACS_CVPR2025/blob/main/figures/Samplingdistribution.png) 




