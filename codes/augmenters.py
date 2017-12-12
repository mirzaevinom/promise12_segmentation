
from __future__ import division, print_function

import cv2
import numpy as np
import SimpleITK as sitk


def elastic_transform(image, x=None, y=None, alpha=256*3, sigma=256*0.07):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    shape = image.shape
    blur_size = int(4*sigma) | 1
    dx = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha
    dy = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha

    if (x is None) or (y is None):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

    map_x =  (x+dx).astype('float32')
    map_y =  (y+dy).astype('float32')

    return cv2.remap(image.astype('float32'), map_y,  map_x, interpolation=cv2.INTER_NEAREST).reshape(shape)


def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs
