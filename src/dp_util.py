import os
import sys
import multiprocessing
import ctypes

import cv2
import torch
import numpy as np


def comToBounds(com, crop_size_3D, fx, fy):
    """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend

        from deep-prior-pp
    """
    if np.isclose(com[2], 0.):
        raise RuntimeError( "Error: CoM ill-defined! This is not implemented")

    zstart = com[2] - crop_size_3D[2] / 2.
    zend = com[2] + crop_size_3D[2] / 2.
    xstart = int(np.floor((com[0] * com[2] / fx - crop_size_3D[0] / 2.) / com[2]*fx+0.5))
    xend = int(np.floor((com[0] * com[2] / fx + crop_size_3D[0] / 2.) / com[2]*fx+0.5))
    ystart = int(np.floor((com[1] * com[2] / fy - crop_size_3D[1] / 2.) / com[2]*fy+0.5))
    yend = int(np.floor((com[1] * com[2] / fy + crop_size_3D[1] / 2.) / com[2]*fy+0.5))
    
    return xstart, xend, ystart, yend, zstart, zend


def getCrop(dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True, background=0):
    """
        Crop patch from image
        :param dpt: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        from deep-prior-pp
    """
    if len(dpt.shape) == 2:
        cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                    abs(yend)-min(yend, dpt.shape[0])),
                                    (abs(xstart)-max(xstart, 0),
                                    abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=background)
    elif len(dpt.shape) == 3:
        cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                    abs(yend)-min(yend, dpt.shape[0])),
                                    (abs(xstart)-max(xstart, 0),
                                    abs(xend)-min(xend, dpt.shape[1])),
                                    (0, 0)), mode='constant', constant_values=background)
    else:
        raise NotImplementedError()

    if thresh_z is True:
        msk1 = np.logical_and(cropped < zstart, cropped != 0)
        msk2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.  # backface is at 0, it is set later
    return cropped

def get2DTransformMatx(xstart, ystart, cropped_shape, out_size_2D):
    # useful for translation matrix for translating y_real_world2d_px coords to relative to CoM
    # doesn't work with data aug
    trans = np.eye(3)
    trans[0, 2] = -xstart
    trans[1, 2] = -ystart
    if cropped_shape[0] > cropped_shape[1]:
        scale = np.eye(3) * out_size_2D[1] / float(cropped_shape[0])
    else:
        scale = np.eye(3) * out_size_2D[0] / float(cropped_shape[1])
    scale[2, 2] = 1

    off = np.eye(3)
    off[0, 2] = 0   # assume new offset == 0 after crop aka no data aug
    off[1, 2] = 0   # assume new offset == 0 after crop aka no data aug

    return np.dot(off, np.dot(scale, trans))

def resizeCrop(crop, sz, interpol_method=cv2.INTER_NEAREST):
    """
        Resize cropped image
        :param crop: crop
        :param sz: size
        :return: resized image
    """
    sz = (int(sz[0]), int(sz[1])) # make sure they are int tuples
    rz = cv2.resize(crop, sz, interpolation=interpol_method)
    return rz

def standardiseImg(depth_img, com3D_mm, crop_dpt_mm, extrema=(-1,1), copy_arr=False):
    # create a copy to prevent issues to original array
    if copy_arr:
        depth_img = np.asarray(depth_img.copy())
    if extrema == (-1,1):
        depth_img[depth_img == 0] = com3D_mm[2] + (crop_dpt_mm / 2.)
        depth_img -= com3D_mm[2]
        depth_img /= (com3D_mm[2] / 2.)
    elif extrema == (0, 1):
        depth_img[depth_img == 0] = com3D_mm[2] + (crop_dpt_mm / 2.)
        depth_img -= (com3D_mm[2] - (crop_dpt_mm / 2.))
        depth_img /= crop_dpt_mm
    else:
        raise NotImplementedError("Please use a valid extrema.")
    
    return depth_img

def standardiseKeyPoints(keypoints_mm, crop_dpt_mm, copy_arr=False):
    '''
        crop_dpt_mm => the z-axis crop length in mm
        returns val in range [-1, 1]
    '''
    ## only one standarisation method, I reckon this is -1 -> 1 standardisation
    ## as by default max values will be -crop3D_sz_mm/2, +crop3D_sz_mm/2
    ## keypoints are the one relative to CoM
    if copy_arr:
        keypoints_mm = np.asarray(keypoints_mm.copy())
    return keypoints_mm / (crop_dpt_mm / 2.)

def unStandardiseKeyPoints(keypoints_std, crop_dpt_mm, copy_arr=False):
    '''
        `keypoints_std` => keypoints in range [-1, 1]
        `crop_dpt_mm` => the z-axis crop length in mm
        returns val in range [-crop3D_sz_mm/2, +crop3D_sz_mm/2]
    '''
    if copy_arr:
        keypoints_std = np.asarray(keypoints_std.copy())
    return keypoints_std * (crop_dpt_mm / 2.)
    
def cropImg(depth_img, com_px, fx, fy, crop3D_mm=(200, 200, 200), out2D_px = (128, 128)):
    """
        from deep-prior
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param crop_size_3D: (x,y,z) extent of the source crop volume in mm
        :param out_size_2D: (x,y) extent of the destination size in pixels
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
    """

    # calculate boundaries in pixels given com in pixel and crop volume in mm
    # conversion is done using principal axis / focal point
    xstart, xend, ystart, yend, zstart, zend = comToBounds(com_px, crop3D_mm, fx, fy)
    
    # crop patch from source
    # crops a 2D image using CoM bounds
    # The x,y bounds are in terms of pixel indices so top-left is 0,0 and 
    # The z bounds are in terms of mm which is pixel value, close pixels have smaller value
    # By default, getCrop thresholds z value suh that all (non-inf depth aka non-zero z)
    # values < min_z_cube is set to min_z_cube
    # all non-zero values > zend are set to 0 aka inf depth
    # so in 3D we can imagine at edge of cube furthest from us all values are 0 (inf dist away)
    # the edge closest to us is max value, nothing comes closer.
    cropped = getCrop(depth_img, xstart, xend, ystart, yend, zstart, zend)
    cropped_resized = resizeCrop(cropped, out2D_px)
    transform_matx = get2DTransformMatx(xstart, ystart, cropped.shape, out2D_px)

    assert(out2D_px[0] == out2D_px[1])    # only 1:1 supported for now

    return cropped_resized, transform_matx


class DeepPriorYTransform(object):
    '''
        Quick transformer for y-vals only (keypoint)
        Centers (w.r.t CoM in dB) and standardises (-1,1) y
        y-val -> center
    '''
    def __init__(self, crop_dpt_mm=200):
        self.crop_dpt_mm = crop_dpt_mm
    
    def __call__(self, sample):
        return \
            standardiseKeyPoints(sample['joints'] - sample['refpoint'], self.crop_dpt_mm).flatten()

class DeepPriorYInverseTransform(object):
    '''
        Quick transformer for y-vals only (keypoint)
        Centers (w.r.t CoM in dB) and standardises (-1,1) y
        
        ## now this is an inverse of above

    '''
    def __init__(self, crop_dpt_mm=200):
        self.crop_dpt_mm = crop_dpt_mm
    
    def __call__(self, sample):
        pred_std_cen_batch, com_batch = sample
        # use ref point to transform back to REAL MM values i.e
        # mm distances of keypoints is w.r.t focal point of img
        ## need to first transform pred_std_centered -> pred_mm_centered using the crop value
        ## then transform pred_mm_centered -> pred_mm_not_centered by appending CoM value
        ## now store this final value in 'keypoints'
        ## also store gt_mm_not_centered in keypoints_gt for future error calc
        
        ## perform the operation in batch .. shuld automatically work with numpy
        #print("\npred_std_cen_batch Shape: ", pred_std_cen_batch.shape)
        #print("\ncom_batch Shape: ", com_batch.shape)
        if len(pred_std_cen_batch.shape) == 3:
            # broadcasting won't work automatically need to adjust array to handle that
            # repetitions will happen along dim=1 so (N, 3) -> (N, 1, 3)
            # Now we can do (N, 21, 3) + (N, 1, 3) as it allows automatic broadcasting along dim=1
            # for (21, 3) + (3,) case this is handled automatically
            com_batch = com_batch[:, None, :]

        return \
            (unStandardiseKeyPoints(pred_std_cen_batch, self.crop_dpt_mm) + com_batch)


# class DeepPriorYInverseTransform(object):
#     '''
#         Quick transformer for y-vals only (keypoint)
#         Centers (w.r.t CoM in dB) and standardises (-1,1) y
#         y-val -> center
#     '''
#     def __init__(self, crop_dpt_mm=200):
#         self.crop_dpt_mm = crop_dpt_mm
    
#     def __call__(self, sample):
#         return \
#             standardiseKeyPoints(sample['joints'] - sample['refpoint'], self.crop_dpt_mm).flatten()



## rename this to deep prior
class DeepPriorXYTransform(object):
    '''
            Deep Prior Transformation class to transform single samples
            by cropping, centering inputs and outputs w.r.t CoM and scaling.

            CoM must be supplied (no calc done) and no augmentations are implemented
    '''
    def __init__(self, depthmap_px=128, crop_len_mm=200):
        self.augmentation = False

        # TODO: make dynamic later as arg of class
        self.fx = 241.42
        self.fy = 241.42
        self.ux = 160.0
        self.uy = 120.0

        self.depthmap_px = depthmap_px
        self.crop_len_mm = crop_len_mm ## aka crop_sz_mm; its one side of a cube

        #self.pca_transformer = pca_transformer

    def __call__(self, sample):
        ### this function is called before reutrning a sample to transform the sample and corresponding
        ### output using a) data augmentation and b) voxelisation
        ### __init__ is called when object is first defined
        ### __call__ is called for all subsequent calls to object
        ## as x,y,z mm where center is center of image
        depth_img = sample['depthmap']  # 2D input image before transformation

        ## equivalent gt3Dorig
        ### ensure this is float32 for consistency as com willbe float32
        ### in future store as float32
        keypoints_gt_mm = sample['joints']

        ## equivalent CoM
        com_refined_mm = sample['refpoint']
        

        ##convert joints to img coords
        keypoints_gt_px = self.joints3DToImg(keypoints_gt_mm)

        ## convert CoM to image coords
        # usually use given CoM
        com_refined_px = self.joint3DToImg(com_refined_mm) #keypoints_gt_px[5]

        # not really need directly use orig com_refined_mm
        # comment this out in future
        # com_refined_mm = self.jointImgTo3D(com_refined_px)


        ## crop3d bounding box size
        ## IMP!!! THIS IS IN MM NOT IN NUM PIXELS!!
        crop_vol_mm = (self.crop_len_mm, self.crop_len_mm, self.crop_len_mm)
        
        out_sz_px = (self.depthmap_px, self.depthmap_px) # in pixels!
        ## convert input image to cropped, centered and resized 2d img
        ## required CoM in px value
        ## convert to 128x128 for network
        ## px_transform_matx for 2D transform of keypoints in px values
        transformed_depth_img, px_transform_matx = cropImg(
                                                         depth_img,
                                                         com_refined_px,
                                                         fx=self.fx, fy=self.fy,
                                                         crop3D_mm=crop_vol_mm,
                                                         out2D_px=out_sz_px)

        ## get keypoints relative to CoM
        ## flatteny to get (21*3,) 1d array
        keypoints_gt_mm_centered = keypoints_gt_mm - com_refined_mm

        # new axis part converts 128x128 -> 1x128x128 for use with deep prior model i.e. each sample is
        # 3D with one channel
        final_depth_img = \
            standardiseImg(transformed_depth_img, com_refined_mm, crop_vol_mm[2], copy_arr=True)[np.newaxis, ...]
        final_keypoints = standardiseKeyPoints(keypoints_gt_mm_centered, crop_vol_mm[2], copy_arr=True).flatten()
        
        #final_pca = self.pca_transformer(final_keypoints)   # transform using pca matx
        #print("Keypoints:", keypoints.shape, "RefPt: ", refpoint.shape, "Pts: ", points.shape)

        ### note final_keypoints are only needed by PCA_model; not by training model
        ### the actual model requires final_depth_img, PCA_outputfor training
        ### GUESS WHAT??? ITS TIME TO ROLL UR OWN PCA MODEL FROM PATTERN REC!

        ### use train_Data to train pattern rec 's pca but must be first collated; supplied to pca
        ### and pca trained


        ### then either save model params if possible but sadly delete train data
        ### now everytime you call transform, u invoke this function then
        # at last line u pass final_keypoints to pca and get 30_component pca vector out!
        ### then thats your final y_output

        
        # return final x, y pair -- HOWEVER FOR TRAINING YOU NEED PCA VERSION OF OUTPUTS!!
        # no need to convert to torch as this is fine as numpy here
        # the dataloader class automatically gets a torch version.

        return (final_depth_img, final_keypoints)

    
    ## from deep-prior
    def jointsImgTo3D(self, sample):
        """
            Normalize sample to metric 3D
            :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
            :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), dtype=np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
            Normalize sample to metric 3D
            :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
            :return: normalized joints in mm
        """
        ret = np.zeros((3,), dtype=np.float32)
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret
    

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), dtype=np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize each joint/keypoint from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joint in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3, ), dtype=np.float32)
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret




class DeepPriorXYTestTransform(DeepPriorXYTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # initialise the super class

    def __call__(self, sample):
        final_depth_img, _ = super().__call__(sample) # transform x, y as usual
        keypoints_gt_mm = sample['joints']  # get the actual y (untransformed)
        com_mm = sample['refpoint'] # neede to transform back output from model

        # basically for test we don't need to transform y_coords
        # hence this derived class is used.
        return (final_depth_img, keypoints_gt_mm, com_mm)


class PCATransform():
    '''
        A simple PCA transformer, supports PCA calc from data matrix and only single sample transformations\n
        `device` & `dtype` => torch device and dtype to use.                            
        `n_components` => Final PCA_components to keep.                         
        `use_cache` => whether to load from cache if exists or save to if doesn't.
        `overwrite_cache` => whether to force calc of new PCA and save new results to disk, 
        overwriting any prev.\n
        PCA is calculated using SVD of co-var matrix using torch (can be GPU) but any subsequent calls
        transform numpy-type samples to new subspace using numpy arrays.
    '''
    def __init__(self, device=torch.device('cpu'), dtype=torch.float,
                 n_components=30, use_cache=False, overwrite_cache=False,
                 cache_dir='checkpoint'):
        
        ## filled on fit, after fit in cuda the np versions copy the matrix and mean vect
        self.transform_matrix_torch = None
        self.transform_matrix_np = None

        self.mean_vect_torch = None
        self.mean_vect_np = None
        
        ## filled on predict
        self.dist_matx_torch = None

        self.device = device
        self.dtype = dtype
        self.out_dim = n_components

        self.use_cache = use_cache
        self.overwrite_cache = overwrite_cache
        self.cache_dir = cache_dir

        if self.use_cache and not self.overwrite_cache:
            self._load_cache()
                
                
                    

    def __call__(self, sample):
        '''
            sample is tuple of 2 np.array (x,y)
            later make this a dictionary
            single y_data sample is 1D
        '''
        if self.transform_matrix_np is None:
            raise RuntimeError("Please call fit first before calling transform.")
        
        #y_data = sample[1]
        
        # automatically do broadcasting if needed, but in this case we will only have 1 sample
        # note our matrix is U.T
        # though y_data is 1D array matmul automatically handles that and reshape y to col vector
        return (sample[0], np.matmul(self.transform_matrix_np, (sample[1] - self.mean_vect_np)))
    
    
    def _load_cache(self):
        cache_file = os.path.join(self.cache_dir, 'pca_'+str(self.out_dim)+'_cache.npz')
        if os.path.isfile(cache_file):
            npzfile = np.load(cache_file)

            matx_shape = npzfile['transform_matrix_np'].shape
            vect_shape = npzfile['mean_vect_np'].shape

            shared_array_base = multiprocessing.Array(ctypes.c_float, matx_shape[0]*matx_shape[1])
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            
            shared_array_base2 = multiprocessing.Array(ctypes.c_float, vect_shape[0])
            shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
            
            shared_array = shared_array.reshape(matx_shape[0], matx_shape[1])
            #print("SharedArrShape: ", shared_array.shape)

            shared_array[:, :] = npzfile['transform_matrix_np']
            shared_array2[:] = npzfile['mean_vect_np']

            self.transform_matrix_np = shared_array
            self.mean_vect_np = shared_array2
        else:
            # handles both no file and no cache dir
            Warning("PCA cache file not found, a new one will be created after PCA calc.")
            self.overwrite_cache = True # to ensure new pca matx is saved after calc
    
    def _save_cache(self):
        ## assert is a keyword not a function!
        assert (self.transform_matrix_np is not None), "Error: no transform matrix to save."
        assert (self.mean_vect_np is not None), "Error: no mean vect to save."

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        cache_file = os.path.join(self.cache_dir, 'pca_'+str(self.out_dim)+'_cache.npz')
        np.savez(cache_file, transform_matrix_np=self.transform_matrix_np, mean_vect_np=self.mean_vect_np)


    
    
    ## create inverse transform function?
    
    def fit(self, X, return_X_no_mean=False):
        ## assume input is of torch type
        ## can put if condition here
        if X.dtype != self.dtype or X.device != self.device:
            X.to(device=self.device, dtype=self.dtype)
        
        if self.transform_matrix_np is not None:
            Warning("PCA transform matx already exists, refitting...")

        # mean normalisation
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)

        # svd, need to transpose x so each data is in one col now
        U,S,_ = torch.svd(torch.t(X))

        print("X.shape:", X.shape, "U.shape:", U.shape)

        ## store U.T as this is the correct matx for single samples i.e. vectors, for multiple i.e. matrix ensure to transpose back!
        self.transform_matrix_torch = torch.t(U[:,:self.out_dim])
        self.mean_vect_torch = X_mean

        # if in cpu just return view tensor as ndarray else copy array to cpu and return as ndarray
        self.transform_matrix_np = self.transform_matrix_torch.cpu().clone().numpy()
        self.mean_vect_np = self.mean_vect_torch.cpu().clone().numpy().flatten() # ensure 1D

        shared_array_base = multiprocessing.Array(ctypes.c_float, self.transform_matrix_np.shape[0]*self.transform_matrix_np.shape[1])
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        
        shared_array_base2 = multiprocessing.Array(ctypes.c_float, self.mean_vect_np.shape[0])
        shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
        
        shared_array = shared_array.reshape(self.transform_matrix_np.shape[0], self.transform_matrix_np.shape[1])
        print("SharedMemArr", shared_array.shape)

        shared_array[:, :] = self.transform_matrix_torch.cpu().clone().numpy()
        shared_array2[:] = self.mean_vect_torch.cpu().clone().numpy().flatten()

        del self.transform_matrix_np
        del self.mean_vect_np

        self.transform_matrix_np = shared_array
        self.mean_vect_np = shared_array2

        self.transform_matrix_np.setflags(write=False)
        self.mean_vect_np.setflags(write=False)

        if self.use_cache and self.overwrite_cache:
            # only saving if using cache feature and allowed to overwrite
            # if initially no file exits overwrite_cache is set to True in _load_cache
            self._save_cache()

        
        if return_X_no_mean:
            # returns the X matrix as mean removed! Also a torch tensor!
            return X
        else:
            return None
    
    
    def fit_transform(self, X):
        ## return transformed features for now
        ## by multiplying appropriate matrix with mean removed X
        # note fit if given return_X_no_mean returns X_no_mean_Torch
        return np.matmul(self.fit(X, return_X_no_mean=True).numpy(), self.transform_matrix_np.T)



class DeepPriorBatchResultCollector():
    def __init__(self, data_loader, transform_output, num_samples):
        self.data_loader = data_loader
        self.transform_output = transform_output
        self.num_samples = num_samples  # need to be exact total calculated using len(test_set)
        
        self.keypoints = None # pred_mm_not_centered
        self.keypoints_gt = None # gt_mm_not_centered
        self.idx = 0
    
    def __call__(self, data_batch):
        ## this function is called when we need to calculate final output error
        
        ### load data straight from disk, y & CoM is loaded straight from the test_loader
        ## x, pred_std_centered, gt_mm_not_centered, CoM = data_batch

        # the first component is input_batch don't need this for now
        # cen => centered i.e. w.r.t CoM
        # nc => not centered i.e. w.r.t focal point of image i.e. similar to gt
        _, pred_std_cen_batch, gt_mm_nc_batch, com_batch = data_batch
        
        pred_std_cen_batch = pred_std_cen_batch.cpu().numpy()
        gt_mm_nc_batch = gt_mm_nc_batch.cpu().numpy()
        com_batch = com_batch.cpu().numpy()

        # an important transformer
        pred_mm_nc_batch = self.transform_output((pred_std_cen_batch, com_batch))

        #print("pred_mm_nc (min, max): (%0f, %0f)\t gt_mm_nc (min, max): (%0f, %0f)" % \
        #        (pred_mm_nc_batch.min(), pred_mm_nc_batch.max(), gt_mm_nc_batch.min(), gt_mm_nc_batch.max()))
        
        ## Note we will have a problem if we have >1 num_batches
        ## and last batch is incomplete, ideally in that case

        if self.keypoints is None:
            # Initialize keypoints until dimensions available now
            self.keypoints = np.zeros((self.num_samples, *pred_mm_nc_batch.shape[1:]))
        if self.keypoints_gt is None:
            # Initialize keypoints until dimensions available now
            self.keypoints_gt = np.zeros((self.num_samples, *gt_mm_nc_batch.shape[1:]))

        batch_size = pred_mm_nc_batch.shape[0] 
        self.keypoints[self.idx:self.idx+batch_size] = pred_mm_nc_batch
        self.keypoints_gt[self.idx:self.idx+batch_size] = gt_mm_nc_batch
        self.idx += batch_size

    def get_result(self):
        ## this will just return predicted keypoints
        return self.keypoints

    
    def calc_avg_3D_error(self, ret_avg_err_per_joint=False):
        ## use self.keypoints for model's results
        ## use self.keypoints_gt for gt results

        ## R^{500, 21, 3} - R^{500, 21, 3} => R^{500, 21, 3} err
        ## R^{500, 21, 3} == l2_err_dist_per_joint ==> R^{500, 21} <-- find euclidian dist btw gt and pred
        err_per_joint = np.linalg.norm(self.keypoints - self.keypoints_gt, ord=2, axis=2)

        ## R^{500, 21} == avg_err_across_dataset ==> R^{21}
        ## do avg for each joint over errors of all samples
        avg_err_per_joint = err_per_joint.mean(axis=0)

        ## R^{21} == avg_err_across_joints ==> R
        avg_3D_err = avg_err_per_joint.mean()

        if ret_avg_err_per_joint:
            return avg_3D_err, avg_err_per_joint
        else:
            return avg_3D_err

        ## for each test frame
        ## calc euclidian dist for each joint's 3D vector between pred and gt to get error matrix
        ## which is R^{21x3}
        ## each row is error of one joint
        ## each col is x,y & z error respectively.
        ## now reduce x,y,z error to single val using euc. dist aka norm
        ## so error is R^{21}
        ## now reduce this dataset matrix of
        ## R^{Nx21} where N is test size to
        ## R^{21} to get avg error of each joint
        ## FINALLY if needed avg R^{21} -> R^{1}
        ## to get avg 3D error