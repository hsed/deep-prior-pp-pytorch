## Structure

### MSRA15 Dataset
[Source](https://www.dropbox.com/s/c91xvevra867m6t/cvpr15_MSRAHandGestureDB.zip?dl=0) ...
[Author](https://jimmysuen.github.io/) ...
[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf)

```
datasets/
    MSRA15/
        -> P0/
            -> 1
                -> 000000_depth.bin
                ...
                -> 000499_depth.bin
                -> joint.txt
            -> 2
            ...
            -> Y
        -> P1
        ...
        -> P8
```

### MSRA15 CoM (Refined)
[Source](https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/center/center.tar.gz) ...
[Repo](https://github.com/dragonbook/V2V-PoseNet-pytorch)

Note these are based on RefinedNet originally implemented in deep-prior-pp, this repo doesn't do refine net implementation

```
datasets/
    MSRA15_CenterofMassPts/
        -> center_test_0_refined.txt
        ...
        -> center_test_8_refined.txt
        -> center_train_0_refined.txt
        ...
        -> center_train_8_refined.txt
```

### Downloading Dataset
```
$> wget -O MSRA15.zip "https://www.dropbox.com/s/c91xvevra867m6t/cvpr15_MSRAHandGestureDB.zip?dl=1"
$> wget -O MSRA15_center.tar.gz "https://cv.snu.ac.kr/research/V2V-PoseNet/MSRA/center/center.tar.gz"
```
