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


### Problem

The problem we face when using the centre of mass from v2v pytorch is that is is given for all frames that are consistent with ahcpe. However ahcpe is not completely the same as dataset.

Thus for test_subj_5:
- `P5/1/000499_depth.bin` and `P5/3/000499_depth.bin` is missing from AHPE msra_test_list.txt.
- This causes the overall P5 test frames to be 8497. This is confirmed in `center_test_5_refined.txt` as its length too is 8597. 
- However, dataset has 8499 frames for test_subj_5. 
- Thus when we reach at missing file `P5/1/000499_depth.bin`, since then all center_test_refined values are essentially WRONG because the order is WRONG.
- Furthermore, because center file is shorter, it causes overall loaded frames to exclude LAST TWO frames (`datasets/MSRA15/P5/Y/000498_depth.bin` and `datasets/MSRA15/P5/Y/000499_depth.bin`) which ARE included in AHPE and our dataset on disk but we can't evaluate it during our testing as it doesn't get loaded.


Solution:
- Directly use the ahpe list of values.
- For each value e.g `P5/1/000498_depth.bin`:
  - Store it directly as filename for imgpath
  - To load centre, use the EXACT index as you enumerate ahpe list of values to load corresponding centre
  - To load gt keypoints, go to that specific folder e.g. 'P5/1' then:
    - Split all lines in joint.txt.
    - Use the filename part e.g. '000498' to index to correct line
    - Copy from there
- Finally, for test_subj_5 you should have 8597 values for TEST_SET ALSO, last two values should correspond to last two values in the `joint.txt` file for the  
