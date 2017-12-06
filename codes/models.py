import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.initializers import RandomNormal, VarianceScaling

def deepest_segmenter(img_rows, img_cols, N=3, k_size=3):

    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**(N), k_size, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool = BatchNormalization()(pool)
    # pool = Dropout(drop)(pool)

    conv2 = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(pool)
    conv2 = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool = BatchNormalization()(pool)
    # pool = Dropout(drop)(pool)

    conv3 = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(pool)
    conv3 = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv3)
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool = BatchNormalization()(pool)
    # pool = Dropout(drop)(pool)

    conv= Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(pool)
    conv= Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(conv)

    conv = Conv2DTranspose(2**(N+2), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv3], axis=3)
    conv = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)
    # conv = Dropout(drop)(conv)


    conv = Conv2DTranspose(2**(N+1), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv2], axis=3)
    conv = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)
    # conv = Dropout(drop)(conv)


    conv = Conv2DTranspose(2**(N), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv1], axis=3)
    conv = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)
    # conv = Dropout(drop)(conv)

    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    model = Model(inputs=[inputs], outputs=[conv])

    return model

def unet_depth_5(img_rows, img_cols, N=3, k_size=3):

    #Encoder part
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(2**(N), k_size, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool = BatchNormalization()(pool)

    conv2 = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(pool)
    conv2 = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool = BatchNormalization()(pool)


    conv3 = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(pool)
    conv3 = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv3)
    pool = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool = BatchNormalization()(pool)

    conv4 = Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(pool)
    conv4 = Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(conv4)
    pool = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool = BatchNormalization()(pool)

    conv5 = Conv2D(2**(N+4), k_size, activation='relu', padding='same' )(pool)
    conv5 = Conv2D(2**(N+4), k_size, activation='relu', padding='same' )(conv5)
    pool = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool = BatchNormalization()(pool)


    conv= Conv2D(2**(N+5), k_size, activation='relu', padding='same' )(pool)
    conv= Conv2D(2**(N+5), k_size, activation='relu', padding='same' )(conv)

    #Decoder Part
    conv = Conv2DTranspose(2**(N+4), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv5], axis=3)
    conv = Conv2D(2**(N+4), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+4), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)


    conv = Conv2DTranspose(2**(N+3), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv4], axis=3)
    conv = Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+3), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(2**(N+2), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv3], axis=3)
    conv = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+2), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(2**(N+1), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv2], axis=3)
    conv = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N+1), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)

    conv = Conv2DTranspose(2**(N), (2, 2), strides=(2, 2), padding='same')(conv)
    conv = concatenate([conv, conv1], axis=3)
    conv = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv)
    conv = Conv2D(2**(N), k_size, activation='relu', padding='same' )(conv)
    conv = BatchNormalization()(conv)


    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    model = Model(inputs=[inputs], outputs=[conv])

    return model


def conv_block(m, dim, acti, bn, res, do=0):

    init = VarianceScaling(scale=1.0/9.0 )
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init )(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n) if bn else n

    return concatenate([n, m], axis=3) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = concatenate([n, m], axis=3)
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m


#Adopted from https://github.com/pietz/unet-keras

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):

	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

if  __name__=='__main__':

    model = UNet((128, 128,1), start_ch=64, depth=3, batchnorm=True, dropout=0.3, residual=True)
    model.summary()
