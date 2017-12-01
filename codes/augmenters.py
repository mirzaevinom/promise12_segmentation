
from __future__ import division, print_function
import numpy as np

#
# from scipy.ndimage.interpolation import map_coordinates
# from scipy.ndimage.filters import gaussian_filter



import SimpleITK as sitk
import cv2

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

    # dx = gaussian_filter((np.random.rand(shape[0],shape[1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # dy = gaussian_filter((np.random.rand(shape[0],shape[1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    if (x is None) or (y is None):
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        # x, y = np.mgrid[0:shape[0],0:shape[1]]
    # indices = np.reshape(x+dx, (-1, 1)), np.reshape( y+dy, (-1, 1) )
    map_x =  (x+dx).astype('float32')
    map_y =  (y+dy).astype('float32')
    # return map_coordinates(image[:,:,0], indices, order=1).reshape(shape)

    return cv2.remap(image.astype('float32'), map_y,  map_x, cv2.INTER_LINEAR).reshape(shape)


def smooth_images(imgs, t_step=0.125, n_iter=5):

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)


    return imgs

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')


    def img_show(img, fname='some.png'):
        img = np.squeeze(img)
        plt.imshow(img, cmap='gray')
        plt.savefig(fname, dpi=300)

    img = np.load('../data/y_test.npy')[20]

    img_show(img, fname='normal.png')
    img_show(elastic_transform(img, alpha=256*1.5), fname='deform.png')


    print()
