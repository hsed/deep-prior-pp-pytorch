if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))   # allow importing from modules in parent dir

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from lib.solver import train_epoch, val_epoch, test_epoch, test_epoch_dropout
from lib.sampler import ChunkSampler
from lib.progressbar import progress_bar, format_time

from datasets.msra_hand import MARAHandDataset
from src.dp_model import DeepPriorPPModel
from src.dp_util import DeepPriorXYTransform, DeepPriorXYTestTransform, \
                        DeepPriorYTransform, DeepPriorYInverseTransform, \
                        DeepPriorBatchResultCollector, PCATransform, \
                        AugType



#######################################################################################
## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    #parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', metavar='EPOCHID', default=-1, type=int,
        help='resume after epoch ID')
    parser.add_argument('--epochs', '-e', metavar='NUMEPOCHS', default=3, type=int,
        help='num epochs (max_epoch_id + 1)')
    parser.add_argument('--device_id', '-d', metavar='GPUID', default=0, type=int,
        choices=range(4), help='GPU Device ID for multi GPU system')
    parser.add_argument('--reduced-dataset', '-rd', action='store_true',
        help='use a reduced dataset, only for testing')
    parser.add_argument('--refined-com', '-rc', action='store_true',
        help='use com from refineNet stored in txt files')
    parser.add_argument('--force-pca', '-fp', action='store_true',
        help='Force new PCA calc and overwrite cache (if exists)')
    args = parser.parse_args()
    return args




def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')





def main():    
    #######################################################################################
    ## Configurations
    #print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda%d' % args.device_id) if torch.cuda.is_available() \
                else torch.device('cpu')
    dtype = torch.float

    #
    args = parse_args()
    resume_train = args.resume >= 0
    resume_after_epoch = args.resume

    save_checkpoint = True
    EPOCHS_PER_CHECKPOINT = 5 # 5
    checkpoint_dir = r'checkpoint'

    START_EPOCH = 0
    NUM_EPOCHS = args.epochs#3#1#2#1

    # Data
    DATA_DIR = r'datasets/MSRA15'
    CENTER_DIR = r'datasets/MSRA15_CenterofMassPts'

    NUM_KEYPOINTS = 21
    TEST_SUBJ_ID = 0 #3 ##changed for now
    PCA_COMP = 30
    IMGSZ_PX = 128 
    CROPSZ_MM = 200

    AUG_MODES = [AugType.AUG_NONE, AugType.AUG_ROT]#[AugType.AUG_ROT] # [None]

    ### if refined_com: TODO: save/load pca with different name!s
    if args.reduced_dataset: print("Warning: Using reduced dataset for training.")
    if args.refined_com: print("Warning: Using refined CoM references for training.")

    ### common kwargs for MSRADataset
    MSRA_KWARGS = {
        'reduce': args.reduced_dataset,
        'use_refined_com': args.refined_com
    }

    ### fix for linux filesystem
    torch.multiprocessing.set_sharing_strategy('file_system')

    ######################################################################################
    ## Transforms
    # use default crop sizes 200mm
    # use overwrite_cache = True when you want to force learn a new PCA matrix
    transform_pca = PCATransform(n_components=PCA_COMP, use_cache=True, overwrite_cache=args.force_pca)

    transform_train = DeepPriorXYTransform(depthmap_px=IMGSZ_PX, crop_len_mm=CROPSZ_MM,
                                           aug_mode_lst=AUG_MODES)
    # its equivalent as train transformer escept for augmentation
    transform_val = DeepPriorXYTransform(depthmap_px=IMGSZ_PX, crop_len_mm=CROPSZ_MM,
                                           aug_mode_lst=[None])
    transform_test = DeepPriorXYTestTransform(depthmap_px=IMGSZ_PX, crop_len_mm=CROPSZ_MM)

    # used for pca_calc
    transform_y = DeepPriorYTransform(crop_dpt_mm=CROPSZ_MM)

    ## used at test time
    transform_output = DeepPriorYInverseTransform(crop_dpt_mm=CROPSZ_MM)

    #######################################################################################
    ## PCA
    # if pca_data wasn't cached we must load all y_data and call fit function
    if transform_pca.transform_matrix_np is None:
        # each sample is 1x21x3 so we use cat to make it 3997x21x3
        # id we use stack it intriduces a new dim so 3997x1x21x3
        # load all y_sample sin tprch array
        # note only train subjects are loaded!
        y_set = MARAHandDataset(DATA_DIR, CENTER_DIR, 'train', TEST_SUBJ_ID, transform_y, **MSRA_KWARGS)
        y_loader = torch.utils.data.DataLoader(y_set, batch_size=1, shuffle=True, num_workers=0)
        print('==> Collating y_samples for PCA ..')
        
        fullYList = []
        for (i, item) in enumerate(y_loader):
            fullYList.append(item)
            progress_bar(i*y_loader.batch_size, len(y_loader))
        
        y_train_samples = torch.cat(fullYList) #tuple(y_loader)
        #print(fullList)
        print("\nY_GT_STD SHAPE: ", y_train_samples.shape, 
                "Max: ", np.max(y_train_samples.numpy()), 
                "Min: ", np.min(y_train_samples.numpy()), "\n")
        # in future just use fit command, fit_transform is just for testing
        print('==> fitting PCA ..')
        zz = transform_pca.fit_transform(y_train_samples)
        print("PCA_Y_SHAPE: ", zz.shape, "MAX: ", zz.max(), "MIN: ", zz.min(), "\n")
        print('==> PCA fitted ..')

        del y_train_samples
        del fullYList
        del y_loader
        del y_set

    # print("PCA Matrix, Vector: \t", 
    #         transform_pca.transform_matrix_np.shape, 
    #         "\t",transform_pca.mean_vect_np.shape)

    ## note if we transpose this matrix its a U matrix so
    ## unitary i.e. inverse is transpose so we can then invert the transformation
    ## so to supply as last layer we will supply transform_matx.T
    #' bias will be supplied as is but may need reshaping.

    #######################################################################################
    ## Data, transform, dataset and loader
    
    print('==> Preparing data ..')
    ## dataset returns np array samples ; dataloader returns torch samples and also shuffled
    train_set = MARAHandDataset(DATA_DIR, CENTER_DIR, 'train', TEST_SUBJ_ID,
                                transforms.Compose([transform_train,transform_pca]), **MSRA_KWARGS)
    ## set here collate_fn as convert to pca
    ## there is some problem here basically we can't parallelize the transformers so 
    ## if u use num_workers=4 it just gives NaN results
    ## if its 0 then all its fine.
    ## i think somehow transformers can't exist in  each thread or something
    ## must be to do with pca function,
    ## shuld we try composing transformers? -- doesnt help
    ## essentially num workers must be 0 -- lets just keep this as is for now
    ## we have fixed this now by using shared_mem types from multiprocessing model
    ## now we run at 100% cpu while training X_X
    # batch size 1 ---> 128
    # 128 -- is from deep_prior_pp
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4) 

    # No separate validation dataset, just use test dataset instead
    # here we are using test_subj_id for val, this is data-snooping
    # so if we do any hyperparam search using valset it will be bad to
    # report test error using same set
    # we'll fix this afterwards...
    # for validation batch size doesn't matter as we don't train, only calc error
    val_set = MARAHandDataset(DATA_DIR, CENTER_DIR, 
                                'test', TEST_SUBJ_ID,
                                transforms.Compose([transform_val,transform_pca]), **MSRA_KWARGS)
    
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=4)#6
    # print("ValSetSz: ", len(val_set))

    ### debugging
    #xTmp, yTmp = train_set[0]
    #quit()
    #print("Y_TMP", yTmp.shape)
    #print(yTmp)
    #quit()
    # we see a time differ of 11s vs 14s for 4 workers vs 0 worker
    # import time
    # t = time.time()
    # a = []
    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #    a.append(targets.clone())

    # c = torch.cat(a)   
    # print("\n\ntook: ", time.time()-t, "s newsamples SHAPE: ", c.shape, "Max: ", np.max(c.numpy()), "Min: ", np.min(c.numpy()))
    # quit()

    #######################################################################################
    ## Model, criterion and optimizer
    #Weight matx is transform_matx.Transpose as its inverse transform
    print('==> Constructing model ..')
    net = DeepPriorPPModel(input_channels=1, num_joints=NUM_KEYPOINTS, 
                           num_dims=3, pca_components=PCA_COMP, 
                           dropout_prob=0.3, train_mode=True,
                           weight_matx_np=transform_pca.transform_matrix_np.T,
                           bias_matx_np=transform_pca.mean_vect_np)

    net = net.to(device, dtype)
    if device == torch.device('cuda'):
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        print('cudnn.enabled: ', torch.backends.cudnn.enabled)

    ## make this better like huber loss etc?
    ## deep-prior-pp in code just uses MSELoss
    criterion = nn.MSELoss()

    ## params from deep-prior-pp
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    #optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)


    #######################################################################################
    ## Resume
    if resume_train:
        # Load checkpoint
        epoch = resume_after_epoch
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')

        print('==> Resuming from checkpoint after epoch id {} ..'.format(epoch))
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

        checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth'))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1


    #######################################################################################
    ## Train and Validate
    print('==> Training ..')
    train_time = time.time()
    # changed training_procedure so that if loaded epoch_id+1 = NUM_EPOCHS then don't train at all
    # i.e. if we are to train 1 epoch and epoch_0 was loaded, no need to train further
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        print('Epoch: {}'.format(epoch))
        train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
        val_epoch(net, criterion, val_loader, device=device, dtype=dtype)

        # if ep_per_chkpt = 5, save as ep_id: 4, 9, 14, 19, 24, 29
        if save_checkpoint and (epoch+1) % (EPOCHS_PER_CHECKPOINT) == 0:
            if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, checkpoint_file)

    print("Training took: ", format_time(time.time() - train_time), '\n')
    #######################################################################################
    ## Test
    print('==> Testing ..')

    # currently our test_set === val_set !! TODO: change this
    # print('Test on test dataset ..')
    test_set = MARAHandDataset(DATA_DIR, CENTER_DIR, 'test', TEST_SUBJ_ID,
                                transform=transform_test, **MSRA_KWARGS)

    ## increaase batch size and workers for fastr (parallel) calc in future
    ## forget batch~_size for now as addition of com_Batch doesnt work properly#
    ## its (500,21,3) + (500,3)
    ## need to use np.repeat to make com (500,3) -> (500,21,3)
    ## ensure error is same regardless of batch size!! --> correct upto 1e-6
    dropout = False

    if dropout:
        test_loader = \
            torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    else:
        test_loader = \
            torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
    test_res_collector = DeepPriorBatchResultCollector(test_loader, transform_output, len(test_set))
    
    if not dropout:
        test_epoch(net, test_loader, test_res_collector, device, dtype)
    else:
        test_epoch_dropout(net, test_loader, test_res_collector, device, dtype)
    #keypoints_test = test_res_collector.get_result()
    # save_keypoints('./test_res.txt', keypoints_test)

    # print('Fit on train dataset ..')
    # fit_set = MARAHandDataset(DATA_DIR, CENTER_DIR, 'train', TEST_SUBJ_ID, transform_test)
    # fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=1, shuffle=False, num_workers=1)
    # fit_res_collector = BatchResultCollector(fit_loader, transform_output)

    # test_epoch(net, fit_loader, fit_res_collector, device, dtype)
    # keypoints_fit = fit_res_collector.get_result()
    # save_keypoints('./fit_res.txt', keypoints_fit)
    print("FINAL_AVG_3D_ERROR: %0.4fmm" % test_res_collector.calc_avg_3D_error())

    print("With Config:", "{GT_CoM: %s, Aug: %s, Full_Dataset: %s}" % \
                (not args.refined_com, False, not args.reduced_dataset ))
    

    print('All done ..')


if __name__ == '__main__':
    main()