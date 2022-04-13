# Conditional image repainting via semantic bridge and piecewise value function (unofficial)

## Introducation
This is the author's unofficial PyTorch implementation.

We study Conditional Image Repainting (CIR) to lower the skill barriers of image editing techniques. 

Conditional content generation refers to visual synthesis tasks conditioned on user inputs. The user inputs cover three aspects, i.e., geometry (semantic parsing mask), colors (attribute and language), and gray-scale textures (latent code).

<!-- ![test image size](https://github.com/shuchenweng/MISC/blob/main/setting.png){:height="50%" width="50%"} -->
 <img src="https://github.com/shuchenweng/TGC/blob/main/birds.png" width = "638" height = "375" alt="图片名称" align=center />

By controlling the language and geomtery inputs, it is easy to generate objects with similar appearance or same shape for our model.

Our model support iterative image editing to modify the color, class or geometry of objects in the wild. After the editing, the whole scenes look quite different, which demonstrates the robustness and flexibility.

<!-- ![test image size](https://github.com/shuchenweng/MISC/blob/main/setting.png){:height="50%" width="50%"} -->
 <img src="https://github.com/shuchenweng/TGC/blob/main/teaser.png" width = "615" height = "376" alt="图片名称" align=center />

## Prerequisites
* Python 3.6
* PyTorch 1.10
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
git clone https://github.com/shuchenweng/TGC.git
```
Install PyTorch and dependencies from http://pytorch.org

Install other python requirements

## Datasets
We process the [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset for evaluation. For CUB, we annotate bird images with parsing masks, and follow AttnGAN for data processing. 

## Getting Started
Download the [proccessed CUB dataset](https://drive.google.com/file/d/1JfQD2wAYYbMPDwD3qTWeAhnIwQ4AI0Oz/view?usp=sharing) and copy them under DATA_DIR.

Download the [pre-trained CUB weights](https://drive.google.com/drive/folders/1c5kAtYvDQxTFvKAmcAiNgY1MqDV82ZVD?usp=sharing) and copy them under PRETRAINED_DIR. 

Setting the MODEL_DIR as the storage directory for generated experimental results.

These directory parameters could be found in cfg/test_bird_SC.yml and cfg/train_bird_SC.yml. 

### 1) Training
```
python main.py --cfg train_bird_SC.yml
```
### 2) Testing
```
python main.py --cfg test_bird_SC.yml
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [Conditional Image Repainting via Semantic Bridge and Piecewise Value Function](https://ci.idm.pku.edu.cn/ECCV20a.pdf)
```
@inproceedings{Repaint,
  title={Conditional image repainting via semantic bridge and piecewise value function},
  author={Weng, Shuchen and Li, Wenbo and Li, Dawei and Jin, Hongxia and Shi, Boxin},
  booktitle={{ECCV}},
  year={2020},
}

```
