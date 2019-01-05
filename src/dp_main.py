if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os

from torchvision import transforms

from lib.solver import train_epoch, val_epoch, test_epoch
from lib.sampler import ChunkSampler

from datasets.msra_hand import MARAHandDataset
from src.dp_model import DeepPriorPPModel
from src.dp_util import DeepPriorXYTransform, DeepPriorYTransform, BatchResultCollector, PCATransform



#######################################################################################
## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    #parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', default=-1, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args




def save_keypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')





def main():    
    keypoints_num = 21
    test_subject_id = 3
    pca_components=30

    # Transform
    # use default crop sizes 200mm
    transform_pca = PCATransform(n_components=pca_components)

    transform_train = DeepPriorXYTransform()
    transform_val = DeepPriorXYTransform()

    transform_y = DeepPriorYTransform()


    #######################################################################################
    ## Configurations
    print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float

    #
    args = parse_args()
    resume_train = args.resume >= 0
    resume_after_epoch = args.resume

    save_checkpoint = True
    checkpoint_per_epochs = 5
    checkpoint_dir = r'checkpoint'

    start_epoch = 0
    epochs_num = 21


    # Data
    data_dir = r'datasets/MSRA15'
    center_dir = r'datasets/MSRA15_CenterofMassPts'

    #######################################################################################
    ## PCA

    #train_set_torch = torch.tensor([1,2,3])
    # Dataset and loader

    y_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_y)
    y_loader = torch.utils.data.DataLoader(y_set, batch_size=1, shuffle=True, num_workers=4)
    # train_loader.__iter__().__next__() ## how it internally works
    
    # each sample is 1x21x3 so we use cat to make it 3997x21x3
    # id we use stack it intriduces a new dim so 3997x1x21x3
    # load all y_sample sin tprch array
    print('==> Collating y_samples for PCA ..')
    y_train_samples = torch.cat(tuple(y_loader))
    # for (i, item) in enumerate(train_loader):
    #     fullList.append(item)
    #     if (i+1) % 500 == 0:
    #         print("500 done")
    #         break
    #print(fullList)
    print("\n\nSAMPLES SHAPE: ", y_train_samples.shape, "Max: ", np.max(y_train_samples.numpy()), "Min: ", np.min(y_train_samples.numpy()))

    print('==> fitting PCA ..')
    zz = transform_pca.fit(y_train_samples)

    print("PCA MAX: ", zz.max(), "MIN: ", zz.min())

    print('==> PCA fitted ..')

    del y_train_samples
    del y_set
    del y_loader

    

    #######################################################################################
    ## Data, transform, dataset and loader
    
    print('==> Preparing data ..')
    ## dataset returns np array samples ; dataloader returns torch samples and also shuffled
    train_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id,
                                transforms.Compose([transform_train,transform_pca]))
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4) #4#6 worker

    # No separate validation dataset, just use test dataset instead
    val_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)#6

    ### debugging
    #xTmp, yTmp = train_loader.__iter__().__next__() #train_set.__getitem__(0) 
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
    print('==> Constructing model ..')
    net = DeepPriorPPModel(input_channels=1, num_joints=21, 
                           num_dims=3, pca_components=30, 
                           dropout_prob=0.3, train_mode=True)

    net = net.to(device, dtype)
    if device == torch.device('cuda'):
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        print('cudnn.enabled: ', torch.backends.cudnn.enabled)

    ## make this better like huber loss etc
    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters())
    #optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)


    #######################################################################################
    ## Resume
    if resume_train:
        # Load checkpoint
        epoch = resume_after_epoch
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')

        print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

        checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth'))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1


    #######################################################################################
    ## Train and Validate
    print('==> Training ..')
    for epoch in range(start_epoch, start_epoch + epochs_num):
        print('Epoch: {}'.format(epoch))
        train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)
        val_epoch(net, criterion, val_loader, device=device, dtype=dtype)

        if save_checkpoint and epoch % checkpoint_per_epochs == 0:
            if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
            checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, checkpoint_file)


    #######################################################################################
    ## Test
    print('==> Testing ..')

    ## transform_output => convert output world3D coords to cropped3d coords
    ## transform_Test ==> same as train?

    print('Test on test dataset ..')
    test_set = MARAHandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test_res_collector = BatchResultCollector(test_loader, transform_output)

    test_epoch(net, test_loader, test_res_collector, device, dtype)
    keypoints_test = test_res_collector.get_result()
    save_keypoints('./test_res.txt', keypoints_test)


    print('Fit on train dataset ..')
    fit_set = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform_test)
    fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=1, shuffle=False, num_workers=1)
    fit_res_collector = BatchResultCollector(fit_loader, transform_output)

    test_epoch(net, fit_loader, fit_res_collector, device, dtype)
    keypoints_fit = fit_res_collector.get_result()
    save_keypoints('./fit_res.txt', keypoints_fit)

    print('All done ..')


if __name__ == '__main__':
    main()