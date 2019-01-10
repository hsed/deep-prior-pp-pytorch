# Deep-Prior-PP-PyTorch
Deep Prior PP implementation in PyTorch

Many util functions based on: https://github.com/moberweger/deep-prior-pp

Many general (and util) functions based on: https://github.com/dragonbook/V2V-PoseNet-pytorch


## Warning (old)
Need to disable cudnn for batchnorm, or just only use cuda instead. With cudnn for batchnorm and in float precision, the model cannot train well. My simple experiments show that:

This was for pytorch 0.4.x, pytorch 1.0 has bug fixed: https://github.com/Microsoft/human-pose-estimation.pytorch/issues/67



## Instructions
All script calls must be made whilst in the root folder location.

```
# for debugging/testing model with random data
python src/dp_model.py

# for actual training using dataset
python src/dp_main.py
```

## Status
Supporting features: 

- [x] MSRA loading + transformer util func
- [x] ResNet-50 based model
- [x] refined CoM support
- [x] training for 'P1' gesture
- [x] training for all gestures (**untested**)
- [x] validation -- (**based on test_set_id only**)
- [x] testing -- based on test_set_id
- [x] avg 3D error calculation on test_set


## Not Implemented
These features are present in original soruce code but not yet implemented here: 
- Data augmentation for training
- Data augmentation for PCA
- CoM detection aka Hand Detector
- RefinedNet (for CoM refining)
- ScaleNet (aka multi-scale training)

## Dataset
See `datasets/README.md` for details on the required datasets.

## Other
See `doc/notes.md` for more details (currently in rough / needs cleanup)