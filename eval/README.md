# Evaluation
## Steps

- For evaluation, train network for each subject id as test case and then in the end choose `--save_eval` to save results (in `eval/MSRA15/` by default) which are particularly compatible with [ahpe](https://github.com/xinghaochen/awesome-hand-pose-estimation/tree/master/evaluation). The file `msra_test_list.txt` is from Awesome-Hand-Pose-Estim. github repo.

- This will save files as:
  ```
  eval/MSRA15/test_0_ahpe.txt
  eval/MSRA15/test_1_ahpe.txt
  .
  .
  .
  eval/MSRA15/test_8_ahpe.txt
  ```

- Now to merge these files run: 
  ```
  $> python lib/eval_merger.py
  ```

- You will now get a single file with concatenated results:
  ```
  eval/MSRA15/eval_all_ahpe_deep_prior_pp_xyz.txt
  ```
- You can convert these to UVD as supplied by ...