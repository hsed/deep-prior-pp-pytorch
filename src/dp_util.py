import sys
import numpy as np
import cv2
import torch

import multiprocessing
import ctypes

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
        crop_dpt_mm => z-axis crop length in mm
    '''
    ## only one standarisation method, I reckon this is -1 -> 1 standardisation
    ## as by default max values will be -crop3D_sz_mm/2, +crop3D_sz_mm/2
    ## keypoints are the centered / translated  one relative to CoM
    if copy_arr:
        keypoints_mm = np.asarray(keypoints_mm.copy())
    return keypoints_mm / (crop_dpt_mm / 2.)
    
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



class PCATransform():
    def __init__(self, device=torch.device('cpu'), dtype=torch.float, n_components=30):
        
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
    
    ## create inverse transform function?

    
    def fit(self, X):
        ## assume input is of torch type
        ## can put if condition here
        if X.dtype != self.dtype or X.device != self.device:
            X.to(device=self.device, dtype=self.dtype)

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
        print("SHapred", shared_array.shape)

        shared_array[:, :] = self.transform_matrix_torch.cpu().clone().numpy()
        shared_array2[:] = self.mean_vect_torch.cpu().clone().numpy().flatten()

        del self.transform_matrix_np
        del self.mean_vect_np

        self.transform_matrix_np = shared_array
        self.mean_vect_np = shared_array2

        # self.transform_matrix_np.flags.writeable = False
        # self.mean_vect_np.flags.writeable = False

        return np.matmul(X.numpy(), self.transform_matrix_np.T)
    
    



class BatchResultCollector():
    def __init__(self, data_loader, transform_output):
        self.data_loader = data_loader
        self.transform_output = transform_output
        self.samples_num = len(data_loader)
        self.keypoints = None
        self.idx = 0
    
    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.cpu().numpy()
        refpoints_batch = extra_batch.cpu().numpy()

        keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

        if self.keypoints is None:
            # Initialize keypoints until dimensions awailable now
            self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

        batch_size = keypoints_batch.shape[0] 
        self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
        self.idx += batch_size

    def get_result(self):
        return self.keypoints