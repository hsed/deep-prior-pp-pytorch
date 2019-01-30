# Deep-Prior-PP-PyTorch
Deep Prior PP implementation in PyTorch

Many util functions based on: https://github.com/moberweger/deep-prior-pp

Many general (and util) functions based on: https://github.com/dragonbook/V2V-PoseNet-pytorch

## Setup
```
$> conda create -n deep-prior python=3.7.1
$> conda activate deep-prior
$> conda install -c anaconda opencv=3.4.2 numpy=1.15.4 matplotlib=3.0.1
$> ### only choose one of the two below
$> conda install -c pytorch pytorch-cpu=1.0.0 torchvision-cpu
$> conda install -c pytorch pytorch=1.0.0 torchvision
```


## Instructions
All script calls must be made whilst in the root folder location.

```
# for debugging/testing model with random data
python src/dp_model.py

# for actual training using MSRA dataset
python src/dp_main.py
```

## Status
### Implemented
Features present in this code: 

- [x] MSRA loading + transformer util func
- [x] ResNet-50 based model
- [x] refined CoM support (**untested**)
- [x] training for 'P1' gesture
- [x] training for all gestures (**untested**)
- [x] validation -- (**based on test_set_id only**)
- [x] testing -- based on test_set_id
- [x] avg 3D error calculation on test_set (using abs dist btw target & output keypoints)
- [x] Data Augmentation for training 
  - [x] Using Rotation
  - [x] Using Translation
  - [x] Using Scaling
- [ ] Data augmentation for PCA

### Not Implemented
These features are present in original soruce code but not yet implemented here:

- 
- CoM detection aka Hand Detector + RefineNet as pipeline
- ScaleNet (aka multi-scale training)
- % Error frames vs max 3D error
- NYU, ICVL datasets

## Results

### Experimented
|CoM| PCA Aug | Train Aug | Error |
|---|---------|-----------|-------|
|RefineNet|None|None|14.6952mm|
|RefineNet|None|Rot+None|13.1496mm|
|RefineNet|None|Scale+None|13.4824mm|
|RefineNet|None|Rot+Trans+None|13.4938mm|
|RefineNet|None|Rot+Scale+None|13.9754mm|


### Target
~9mm with PCA augmentation + rot+scale+trans augmentation for MSRA dataset

## Dataset
See `datasets/README.md` for details on the required datasets.

## Other
See `doc/notes.md` for more details (currently in rough / needs cleanup)

[Progress Doc](https://imperiallondon-my.sharepoint.com)
