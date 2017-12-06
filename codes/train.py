#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
from collections import defaultdict
import os, pickle, sys
import shutil
from functools import partial
from itertools import izip

import cv2
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist

from models import *
from metrics import dice_coef, dice_coef_loss
from augmenters import *

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def data_to_array(img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)) )

    fileList =  os.listdir('../data/train/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList.sort()

    val_list = [5,15,25,35,45]
    train_list = list( set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

        for filename in filtered:

            itkimage = sitk.ReadImage('../data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )

            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs )

        images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = masks.astype(int)

        #Smooth images using CurvatureFlow
        images = smooth_images(images)

        if count==0:
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save('../data/X_train.npy', images)
            np.save('../data/y_train.npy', masks)
        elif count==1:
            images = (images - mu)/sigma

            np.save('../data/X_val.npy', images)
            np.save('../data/y_val.npy', masks)
        count+=1

    fileList =  os.listdir('../data/test/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList.sort()
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage('../data/test/'+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append( len(imgs) )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    images = smooth_images(images)
    images = (images - mu)/sigma
    np.save('../data/X_test.npy', images)
    np.save('../data/test_n_imgs.npy', np.array(n_imgs) )


def load_data():

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')


    return X_train, y_train, X_val, y_val


def create_val_from_train(X_train, y_train, seed=10):

    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )
    # we create two instances with the same arguments
    data_gen_args = dict(preprocessing_function=elastic)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=100, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=100, seed=seed)

    train_generator = izip(image_generator, mask_generator)

    count=0
    X_val = []
    y_val = []

    for X_batch, y_batch in train_generator:

        if count==5:
            break

        count+=1

        X_val.append(X_batch)
        y_val.append(y_batch)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    return X_val, y_val

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.32
    epochs_drop = 5
    lrate = initial_lrate * drop**int((1 + epoch) / epochs_drop)
    return lrate


def keras_fit_generator(img_rows=96, img_cols=96, n_imgs=10**4, batch_size=32, regenerate=True):

    if regenerate:
        data_to_array(img_rows, img_cols)
        #preprocess_data()

    X_train, y_train, X_val, y_val = load_data()
    # X_val, y_val = create_val_from_train(X_val, y_val, seed=10)
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    # Provide the same seed and keyword arguments to the fit and flow methods

    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )
    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=45.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[1, 1.2],
        fill_mode='constant',
        preprocessing_function=elastic)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2
    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(y_train, seed=seed)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = izip(image_generator, mask_generator)

    model = UNet((img_rows, img_cols,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, residual=True)
    # model.load_weights('../data/weights.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        '../data/weights.h5', monitor='val_loss', save_best_only=True)

    c_backs = [model_checkpoint]

    #c_backs.append( LearningRateScheduler(step_decay) )
    c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )

    model.compile(  optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])

    model.fit_generator(
                        train_generator,
                        steps_per_epoch=n_imgs//batch_size,
                        epochs=20,
                        verbose=2,
                        shuffle=True,
                        validation_data=(np.concatenate([X_train,X_val]), np.concatenate([y_train,y_val]) ),
                        callbacks=c_backs,
                        use_multiprocessing=True)



if __name__=='__main__':

    import time

    start = time.time()
    keras_fit_generator(img_rows=256, img_cols=256, regenerate=False,
                        n_imgs=3*10**4, batch_size=32)

    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2 ) )
