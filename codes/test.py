#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
"""

from __future__ import division, print_function
import numpy as np
print()

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.optimizers import Adam

from models import *
from metrics import *
import os
import matplotlib.gridspec as gridspec
import SimpleITK as sitk

def make_plots(img, segm, segm_pred):
    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    from skimage.measure import find_contours


    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm], cmap='gray' )
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')


        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1
        # if mm==0:
        #     ax.set_title('MRI image', fontsize=20)
        # ax = fig.add_subplot(gs[n_cols*mm+1])
        # ax.imshow(segm[mm], cmap='gray', vmin=0, vmax=1 )
        # ax.axis("off")  # remove axis
        # ax.set_aspect(1)  # aspect ratio of 1
        # if mm==0:
        #     ax.set_title('Ground Truth', fontsize=20)
        #
        # ax = fig.add_subplot(gs[n_cols*mm+2])
        # ax.imshow(segm_pred[mm], cmap='gray' , vmin=0, vmax=1 )
        # ax.axis("off")  # remove axis
        # ax.set_aspect(1)  # aspect ratio of 1
        # if mm==0:
        #     ax.set_title('Prediction', fontsize=20)
    return fig


def check_predictions(n_best=20, n_worst=20 ):
    if not os.path.isdir('../images'):
        os.mkdir('../images')

    X_test = np.load('../data/X_test.npy')
    y_test = np.load('../data/y_test.npy')

    img_rows = X_test.shape[1]
    img_cols = img_rows

    # model = unet_w_batchnorm(img_rows, img_cols, N=4)
    model = UNet((img_rows, img_cols,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5 )
    model.load_weights('../data/weights.h5')

    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])

    y_pred = model.predict( X_test, verbose=1)

    # from scipy import ndimage
    #
    # for mm in range(len(y_pred)):
    #     y_pred[mm,:,:,0] = ndimage.binary_fill_holes(y_pred[mm,:,:,0]).astype(int)

    axis = tuple( range(1, y_test.ndim ) )
    scores = numpy_dice( y_test, y_pred , axis=axis)
    print('Mean DSC:', scores.mean() )
    print('Median DSC:', np.median(scores) )
    print('Max DSC:', scores.max() )
    print('Test accuracy:',numpy_dice( y_test, y_pred , axis=None))

    n_imgs = n_imgs_case().cumsum()
    vol_scores = np.zeros( len(n_imgs) )
    ravd = np.zeros( len(n_imgs) )
    for mm in range(len(n_imgs)):
        if mm==0:
            ravd[mm] = rel_abs_vol_diff( y_test[:n_imgs[mm] ] , y_pred[:n_imgs[mm]] )
            vol_scores[mm] = numpy_dice( y_test[:n_imgs[mm] ] , y_pred[:n_imgs[mm]] , axis=None)
        else:
            ravd[mm] = rel_abs_vol_diff( y_test[n_imgs[mm-1]:n_imgs[mm] ] , y_pred[n_imgs[mm-1]:n_imgs[mm]] )
            vol_scores[mm] = numpy_dice( y_test[n_imgs[mm-1]:n_imgs[mm] ] , y_pred[n_imgs[mm-1]:n_imgs[mm]] , axis=None)

    print('Mean volumetric DSC:', vol_scores.mean() )
    print('Mean Rel. Abs. Vol. Diff:', ravd)

    #PLotting the results
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y_test.sum(axis=axis) )[0]

    #Add some best and worst predictions
    img_list = []
    count = 1
    for ind in sort_ind:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X_test[img_list].reshape(-1,img_rows, img_cols)
    segm = y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    np.save('../data/y_pred.npy', y_pred)

    fig = make_plots(img, segm, segm_pred)
    fig.savefig('../images/best_predictions.png', bbox_inches='tight', dpi=300 )


    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X_test[img_list].reshape(-1,img_rows, img_cols)
    segm = y_test[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    fig = make_plots(img, segm, segm_pred)
    fig.savefig('../images/worst_predictions.png', bbox_inches='tight', dpi=300 )


def n_imgs_case():

    test_list = [5,15,25,35,45]
    fileList =  os.listdir('../data/train/')
    fileList = filter(lambda x: '.mhd' in x, fileList)

    filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in test_list), fileList)
    n_imgs = []

    for filename in filtered:

        if 'segm' in filename.lower():
            itkimage = sitk.ReadImage('../data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)
            n_imgs.append( len(imgs) )

    return np.array(n_imgs)


if __name__=='__main__':

    check_predictions( )
    # plt.show()
