
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.initializers import RandomNormal, VarianceScaling
import numpy as np

#Adopted from https://github.com/pietz/unet-keras
#Added kernel initializers based on VarianceScaling
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


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):

	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)

if  __name__=='__main__':

    model = UNet((256, 256,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, upconv=True, maxpool=True, residual=True)
    model.summary()
