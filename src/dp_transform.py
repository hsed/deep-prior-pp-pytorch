import cv2
import torch
import numpy as np


'''
    Variable naming scheme:
    <keypt|com>_<px|mm>_<type1>_<type2>

    dpt_<type1>_<type2>

    type1 => <orig|crop>
    type2 => <BLANK|aug>

    e.g. com_px_crop_aug => 
            com in px coords (last_dim =1 or 'z') w.r.t cropped & augmented depth img

    <transform>Hand2D => All functions returning transformed depth+keypt+com
    <transform>Depth2D => All functions returning transformed depth omly
    <transform>KeyPt2D => All functions returning transformed keypt only
'''

def comToBounds(com, crop_size_3D, fx, fy):
    """
        Project com in px coord to 3D coord and then crop a 3D 'volume' region 
        defined by crop_size_3D, with com at center and then backproject com to
        px coord
        `com` center of mass, in image coordinates (x,y,z), z in mm
        `size` (x,y,z) extent of the source crop volume in mm
        `return` xstart, xend, ystart, yend, zstart, zend as px idx to crop

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

def cropDepth2D(depth_img, com_px, fx, fy, crop3D_mm=(200, 200, 200), out2D_px = (128, 128)):
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

    return np.array(np.dot(off, np.dot(scale, trans)), dtype=np.float32)


def affineTransform2D(M, X):
    '''
        M must be a (m, 3) matx with
            m => {2, 3} only
        X can be a 
            (N,) 1D matx (vector) OR
            (N,k) 2D matx with
                k => {2, 3} only

    '''
    ## in order to perform transl we need a "homogenous vector" which is a vector
    ## we can actually transl using (2x3) matrix but then the operation is non-linear
    ## as simply there is no inverse for (2x3) matx (maybe psuedo inverse...)
    ## written in homogenous coord as opposed to cartesian coords
    ## its for projective geometry and requires a 2D vector as [x,y,1]^T
    ## here we transform com using how many 'pixels' it must move left and down
    ## along the 2D plane to now be the center of the image.
    ## we can imagine a crop area around the hand and then that being 
    ## translated towards the center of orig img area
    ## where the crop area < orig img area
    
    if M.shape == (2, 3):
        # special case if matx comes from opencv
        M = np.vstack((M, np.array([0,0,1], dtype=np.float32)))
    elif M.shape != (3, 3):
        raise AssertionError("Error: Matrix M must be 2x3 or 3x3") 

    if len(X.shape) == 2:
        ## X is 2D matx, assume each sample is row vector in this matx
        if X.shape[1] == 2 or X.shape[1] == 3:
            # for affine transform must use homogeneous coords
            # this condition works for 2D row vector or 3D
            # Y tries to keep orig shape
            Y = np.matmul(np.column_stack((X[:,:2], np.ones(X.shape[0], dtype=np.float32))), M.T)

            ## after multiply Y will always be of shape (m, 3)
            ## if each row in X is 2-elem then this function would make Y (m, 2)
            ## otherwise it ensures that the third elem is orig 'z' value
            ## in X, untransformed.
            Y = np.column_stack((Y[:,:2], X[:, 2])) if X.shape[1] == 3 else Y[:,:2]
        else:
            raise AssertionError("Error: X.shape[1] must be 2 or 3")
    elif len(X.shape) == 1:
        ## X is a 1D vector
        if X.shape[0] == 2 or X.shape[0] == 3:
            Y = np.matmul(M, np.array([X[0], X[1], 1], dtype=np.float32))
            Y = np.array([Y[0], Y[1], X[2]]) if X.shape[0] == 3 else Y[:2]
        else:
            raise AssertionError("Error: X.shape[0] must be 2 or 3")
    else:
        raise AssertionError("Error: X.shape[1] must be 1 or 2")
    
    return Y


def affineTransformImg(M, dpt, resize_method=cv2.INTER_NEAREST,
                       border_mode=cv2.BORDER_CONSTANT, pad_value=0):
    '''
        Can supply 2x3 or 3x3 matrix for M
        if 3x3, it is converted to 2x3 using upper two rows
    '''
    if M.shape[0] == 2 or M.shape[0] == 3:
        return cv2.warpAffine(dpt, M[:2,:], (dpt.shape[1], dpt.shape[0]), flags=resize_method,
                                    borderMode=border_mode, borderValue=pad_value)
    else:
        raise AssertionError("Error: M must be of shape (2,3) or (3,3)")


def cropHand2D(dpt_orig, keypt_px_orig, com_px_orig, fx, fy,
                crop3D_mm=(200, 200, 200),  out2D_px = (128, 128)):
    '''
        Returns depth, keypt & com all cropped & centered in 2D plane as px values
        Also returns 2D affine transform matx.

        This function is only useful to visually plot crops.
        In reality we would just need cropDepth2D as the keypt&com coords
        need for actual output have to be in mm and this is evaluated by
        keypt_mm_crop = keypt_mm_orig - com_mm_orig
    '''

    dpt_crop, crop_transf_matx = cropDepth2D(dpt_orig, com_px_orig, fx, fy, crop3D_mm, out2D_px)

    # note after calling this z-value is always maintained
    keypt_px_crop = affineTransform2D(crop_transf_matx, keypt_px_orig)
    com_px_crop = affineTransform2D(crop_transf_matx, com_px_orig)

    return dpt_crop, keypt_px_crop, com_px_crop, crop_transf_matx


def rotateHand2D(dpt, keypt_px, com_px, rot,
               resize_method=cv2.INTER_NEAREST,
               pad_value=0):
        """
        Rotate hand virtually in the image plane by a given angle
        Only does 2D plane rotation so all inputs must be w.r.t px coords
        :param dpt: Can be either dpt_orig or dpt_crop
        :param keypt_px: Can be either keypt_px_orig or keypt_px_crop
        :param crop_px: Can be either crop_px_orig or crop_px_crop
        :param rot: rotation angle in deg, +ve => acw rotation
        :return: dpt_<orig|crop>_aug, keypt_<orig|crop>_aug, com_<orig|crop>_aug, rot_matx

        Edited from DeepPrior
        """

        # if rot is 0, nothing to do
        if np.allclose(rot, 0.):
            return dpt, keypt_px, com_px, np.eye(3, dtype=np.float32)
        
        # note in deep-prior '-rot' is used for rotation matx s.t. if rot > 0 => cw rot
        # we use 'rot' like standard so now if rot > 0 => acw rot
        rot = np.mod(rot, 360)
        M = cv2.getRotationMatrix2D((dpt.shape[1]//2, dpt.shape[0]//2), rot, 1)
        M = M.astype(np.float32) # make sure everything is float32

        dpt_aug = affineTransformImg(M, dpt, resize_method=resize_method,
                                     border_mode=cv2.BORDER_CONSTANT, pad_value=pad_value)

        # dpt_aug = cv2.warpAffine(dpt, M, (dpt.shape[1], dpt.shape[0]), flags=flags,
        #                          borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value)

        com_px_aug = affineTransform2D(M, com_px)
        keypt_px_aug = affineTransform2D(M, keypt_px)

        return dpt_aug, keypt_px_aug, com_px_aug, M


def scaleHand2D(dpt, keypt_px, com_px, sf,
               resize_method=cv2.INTER_NEAREST,
               pad_value=0):
        pass
    

def translateHand2D(dpt_orig, keypt_px, com_px, keypt_mm, com_mm, off, fx, fy,
               resize_method=cv2.INTER_NEAREST, pad_value=0, dpt_crop_shape=(128,128)):
        if np.allclose(off, 0.):
            return dpt_orig, keypt_px, com_px, np.eye(3, dtype=np.float32)

        com_mm_aug = com_mm + off

        if not (np.allclose(com[2], 0.) or np.allclose(new_com[2], 0.)):
            ### main idea,,, we need to simply recrop the original image 
            ### but this time with new com value
            ### also in the end transform using appropriate transform the keypt_mm
            ### you can do this using M matrix after crop
            ### notice your com WILL ALWAYS BE at center
            ### so if simply you choose a different com, your hand will pivot about
            ### that value and thus you will have a 'translated image'
            ### the only problem is now that our augmentation matrix is a combination
            ### of tw transforms

            ### Need M_aug
            dpt_crop_aug, crop_aug_transf = \
                cropDepth2D(dpt_orig, com_px_orig, fx=fx, fy=fy, 
                            crop3D_mm=crop_vol_mm, out2D_px=out_sz_px)           
        else:
            Mnew = np.eye(3, dtype=np.float32)
            new_dpt = dpt