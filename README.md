# Deep-Prior-PP-PyTorch
Deep Prior PP implementation in PyTorch

Many util functions based on https://github.com/moberweger/deep-prior-pp
Many general (and util) functions based on https://github.com/dragonbook/V2V-PoseNet-pytorch


## Warning
Need to disable cudnn for batchnorm, or just only use cuda instead. With cudnn for batchnorm and in float precision, the model cannot train well. My simple experiments show that:

This was for pytorch 0.4.x, pytorch 1.0 has bug fixed: https://github.com/Microsoft/human-pose-estimation.pytorch/issues/67



## Instructions

```
python src/dp_model.py # for sample data model training

python srtc/dp_main.py # for actual training
```

## Status
Supporting features: 

- [x] MSRA loading + transformer util func
- [x] refined CoM support
- [x] training for 'P1' gesture
- [x] training for all gestures (**untested**)
- [x] validation -- (**based on test_set_id only**)
- [x] testing -- based on test_set_id
- [x] avg 3D error calculation on test_set


See doc/notes.md for more details (currently in rough / needs cleanup)