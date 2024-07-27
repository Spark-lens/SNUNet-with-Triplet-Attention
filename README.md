# SNUNet-with-Triplet-Attention
Remote Sensing Image Change Detection in SNUNet-CD Network Based on Triplet Attention

## Requirements

```python
Python 3.10
Pytorch 2.3.1
torchvision 0.18.1

# recommended to use conda
conda install opencv-python tqdm tensorboardX scikit-learn
pip install opencv-python tqdm tensorboardX scikit-learn
```

## Dataset

[CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)、[LEVIR-CD](https://chenhao.in/LEVIR/)、[WHU](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

## Model

[BIT_CD](https://github.com/justchenhao/BIT_CD)、[CGNet-CD](https://github.com/ChengxiHAN/CGNet-CD)、[SNUNet-CD](https://github.com/likyoo/Siam-NestedUNet)

## Attention

[FCANet](https://github.com/cfzd/FcaNet)、[CA](https://github.com/houqb/CoordAttention)、[Triplet Attention](https://github.com/landskape-ai/triplet-attention)

## Dataset path modification

```python
metadata.json
# Dataset path
"dataset_dir": "../CDD/Real/subset/"
```

## Training model

```python
python train.py 
```

### Use the weights in the weight directory to get visualization results

```python
# Visualization
path = './weights/snunet-32_TA.pt'  # Model path
python visualization.py 

# Model explanation
snunet-32_TA                                        # Improved model weights
snunet-32_ori、snunet-40_ori、snunet-48_ori		# SNUNet model training weights
```

Due to the size limitation of the storage repository, the weights directory only contains the snunet-32_TA weight. Other weights can be downloaded and used separately.

[weights download](https://drive.google.com/drive/folders/1qIlzGXPBTC8b8jPQ2SMX4HreB5Q1wKXy)

