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
import cv2
import os
import matplotlib.gridspec as gridspec
import SimpleITK as sitk
from skimage.transform import resize
from skimage.measure import find_contours

def make_plots(X, y, y_pred, n_best=20, n_worst=20):
    #PLotting the results'
    img_rows = X.shape[1]
    img_cols = img_rows
    axis =  tuple( range(1, X.ndim ) )
    scores = numpy_dice(y, y_pred, axis=axis )
    sort_ind = np.argsort( scores )[::-1]
    indice = np.nonzero( y.sum(axis=axis) )[0]
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
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    img_list = []
    count = 1
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')


    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
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

    fig.savefig('../images/worst_predictions.png', bbox_inches='tight', dpi=300 )

def predict_test(folder='../data/test/', dest='../data/predictions'):
    if not os.path.isdir(dest):
        os.mkdir(dest)

    X_test = np.load('../data/X_test.npy')
    n_imgs = np.load('../data/test_n_imgs.npy').cumsum()

    img_rows = X_test.shape[1]
    img_cols = img_rows
    model = get_model(img_rows, img_cols)
    y_pred = model.predict(X_test, verbose=1, batch_size=128)

    fileList =  os.listdir(folder)
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList.sort()

    start_ind=0
    end_ind=0

    for filename in fileList:
        itkimage = sitk.ReadImage(folder+filename)
        img = sitk.GetArrayFromImage(itkimage)
        start_ind = end_ind
        end_ind +=len(img)
        pred = resize_pred_to_val( y_pred[start_ind:end_ind], img.shape )
        pred = np.squeeze(pred)
        mask = sitk.GetImageFromArray( pred)
        mask.SetOrigin( itkimage.GetOrigin() )
        mask.SetDirection( itkimage.GetDirection() )
        mask.SetSpacing( itkimage.GetSpacing() )
        sitk.WriteImage(mask, dest+'/'+filename[:-4]+'_segmentation.mhd')

def plot_test_samples():

    fileList = os.listdir('../data/test_samples')
    fileList = filter(lambda x: '.png' in x, fileList)
    fileList.sort()
    case_slices = [ [int(s) for s in fname.replace('.', '_').split('_') if s.isdigit()] for fname in fileList]
    case_slices = np.array(case_slices)
    X_test = np.load('../data/X_test.npy')
    n_imgs = np.load('../data/test_n_imgs.npy').cumsum()

    img_rows = X_test.shape[1]
    img_cols = img_rows
    model = get_model(img_rows, img_cols)

    n_cols= 4
    n_rows = int( np.ceil(len(case_slices)/n_cols*2) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    imgs = [ X_test[n_imgs[row[0]-1]+row[1]] for row in case_slices ]
    imgs = np.stack(imgs)
    masks = model.predict( imgs, verbose=1)
    for mm, row in enumerate(case_slices):
        ax = fig.add_subplot(gs[2*mm])
        img = cv2.imread('../data/test_samples'+'/'+fileList[mm],cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
        ax = fig.add_subplot(gs[2*mm+1])
        ax.imshow(imgs[mm,:,:,0], cmap='gray' )
        contours = find_contours(masks[mm,:,:,0], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')

    fig.savefig('../images/test_samples.png', bbox_inches='tight', dpi=300 )


def get_model(img_rows, img_cols):
    model = UNet((img_rows, img_cols,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
    model.load_weights('../data/weights.h5')
    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model

def check_predictions(the_list, plot=False ):

    if not os.path.isdir('../images'):
        os.mkdir('../images')

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')

    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')

    img_rows = X_val.shape[1]
    img_cols = img_rows

    model = get_model(img_rows, img_cols)

    if len(the_list)>10:
        y_pred = model.predict( X_train, verbose=1,batch_size=128)
        print('Results on train set:')
        print('Accuracy:', numpy_dice(y_train, y_pred))

    else:
        y_pred = model.predict( X_val, verbose=1,batch_size=128)
        print('Results on validation set')
        print('Accuracy:', numpy_dice(y_val, y_pred))

    vol_scores = []
    ravd = []
    scores = []
    hauss_dist = []
    mean_surf_dist = []

    start_ind = 0
    end_ind  = 0
    for y_true, spacing in read_cases(the_list):

        start_ind = end_ind
        end_ind +=len(y_true)

        y_pred_up = resize_pred_to_val( y_pred[start_ind:end_ind], y_true.shape)

        ravd.append( rel_abs_vol_diff( y_true , y_pred_up ) )
        vol_scores.append( numpy_dice( y_true , y_pred_up , axis=None) )
        surfd = surface_dist(y_true , y_pred_up, sampling=spacing)
        hauss_dist.append( surfd.max() )
        mean_surf_dist.append(surfd.mean())
        axis = tuple( range(1, y_true.ndim ) )
        scores.append( numpy_dice( y_true, y_pred_up , axis=axis) )

    ravd = np.array(ravd)
    vol_scores = np.array(vol_scores)
    scores = np.concatenate(scores, axis=0)

    print('Mean volumetric DSC:', vol_scores.mean() )
    print('Median volumetric DSC:', np.median(vol_scores) )
    print('Std volumetric DSC:', vol_scores.std() )
    print('Mean Hauss. Dist:', np.mean(hauss_dist) )
    print('Mean MSD:', np.mean(mean_surf_dist) )
    print('Mean Rel. Abs. Vol. Diff:', ravd.mean() )

    if plot and len(the_list)>10:
        make_plots(X_train, y_train, y_pred)
    elif plot:
        make_plots(X_val, y_val, y_pred)

def resize_pred_to_val(y_pred, shape):
    row = shape[1]
    col =  shape[2]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm,:,:] =  cv2.resize( y_pred[mm,:,:,0], (row, col), interpolation=cv2.INTER_NEAREST)

    return resized_pred.astype(int)

def read_cases(the_list=None, folder='../data/train/', masks=True):
    fileList =  os.listdir(folder)
    fileList = filter(lambda x: '.mhd' in x, fileList)
    if masks:
        fileList = filter(lambda x: 'segm' in x.lower(), fileList)
    fileList.sort()
    if the_list is not None:
        fileList = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

    for filename in fileList:
        itkimage = sitk.ReadImage(folder+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        yield imgs, itkimage.GetSpacing()[::-1]



if __name__=='__main__':

    val_list = [5,15,25,35,45]
    train_list = list( set(range(50)) - set(val_list ) )
    train_list.sort()
    # check_predictions( val_list, plot=False)
    # check_predictions( train_list, plot=False)
    # predict_test()
    plot_test_samples()
