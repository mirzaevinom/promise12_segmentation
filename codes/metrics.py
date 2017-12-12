from __future__ import division, print_function

from keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology

def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )


def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)


def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)



def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)


    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds
