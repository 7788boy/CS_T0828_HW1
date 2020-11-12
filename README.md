# Project: Image classification

## Overview
The target of this homework is image classification.  
Training 11185 pictures with cars in different type and 196 labels.   
I use EfficientNet as pretrained model and training with 300 epochs.  

## Download Official Image
You need to download the training and testing dataset from Kaggle.  
https://www.kaggle.com/c/cs-t0828-2020-hw1/data  

## Installation
*Pytorch 1.7.0
*Numpy 1.19.2
*Torchvision 0.8.1
*Tensorboard 2.3.0
*Tqdm 4.51.0
*Cuda 10.1
*Efficientet
```
pip install keras_efficientnets
pip install efficientnet_pytorch
``` 
## Usage
Run train.py to start training. You can set dirfferent parameter by the following command.  
```
python train.py

--h
--mode
--epochs
--batch
--ckptID
```

Test the model  
```
python train.py --mode test --ckptID <your checkpoint ID>
```

Check training log  
```
tensorboard --logdir=runs
```
## Reference
https://arxiv.org/abs/1905.11946  
https://zhuanlan.zhihu.com/p/102467338  
https://pytorch.org/docs/stable/torchvision/models.html#classification  
https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/  

###### tags: `Image classification` `Deep learning` `NCTU CS` `EfficientNet`
