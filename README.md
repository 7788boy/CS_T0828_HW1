# Project:Image classification

## Overview
The target of this homework is image classification.  
Training 11185 pictures with cars in different type and 196 labels.   I use EfficientNet as pretrained model and training with 200 epochs.  

## Download dataset
You need to download the training and testing dataset from Kaggle.  
https://www.kaggle.com/c/cs-t0828-2020-hw1/data  

## Install
Pytorch
Numpy
tqdm
csv
PIL
```
pip install keras_efficientnets
pip install efficientnet_pytorch
```
create dir checkpoints to store checkpoint  

## Usage
```
python train.py
```
```
--h
--mode
--epochs
--batch
--ckptID
```
```
tensorboard --logdir=runs
```
## Performance

## Reference
https://arxiv.org/abs/1905.11946  
https://zhuanlan.zhihu.com/p/102467338  
https://pytorch.org/docs/stable/torchvision/models.html#classification  
https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/  

###### tags: `Image classification` `Deep learning` `NCTU CS` `EfficientNet`
