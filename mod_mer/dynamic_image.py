
'''
Only the function `make_dyn_img` should be used by other modules,
as that is the only intended top-level function.

Rest of the functions needn't be imported, as they can only be called
in certain situations, which only exist within this module.

Note:
This module  is a Python implementation of the original Matlab code
found in the following file:
DynamicImage.m
    from
    http://users.cecs.anu.edu.au/~basura/dynamic_images/code.zip

And the .zip file was found here:
    https://github.com/hbilen/dynamic-image-nets/blob/master/dicnn/compute_approximate_dynamic_images.m
    Line 7

'''

import os
import time
import re
from pprint import pprint

import cv2
import numpy as np
from numpy.linalg import norm
from numpy import matlib
from sklearn.svm import LinearSVR


__all__ = (
    'make_dyn_img',
)


CVAL = 10

# https://bitbucket.org/bfernando/videodarwin/src/master/VideoDarwin.m
# CVAL = 1 

def get_image_paths(folder_with_images):
    s = os.listdir(folder_with_images)

    # Sort the image paths based on the image number only
    s.sort(key=lambda x : int(re.search(r'\d+', x).group()))

    res = []
    for si in s:
        res.append(os.path.join(folder_with_images, si))
    
    return res

def make_4darray(list_paths):
    '''
    Make 4d array of image paths as (timestep, height, width, rgb)

    Note: this function assumes all the images have the same height and width
    '''

    timesteps = len(list_paths)
    i0 = cv2.imread(list_paths[0])
    h, w, rgb = i0.shape
    del i0

    np4d = np.zeros(shape=(timesteps, h, w, rgb), dtype='uint8')

    for index, img in enumerate(list_paths):
        np4d[index] = cv2.imread(img)
    
    return np4d


KWSET = {
    # 'zMean', 
    # 'zMax', 
    'zWF', 
    # 'zWR', 
    # 'FirstFrame'
}
def get_dynamic_image(video4d, image_prefix, out_directory):
    '''
    video4d: 4d nd array of video (t, h, w, d)
    image_prefix: prefix to give to image when saving it
    out_directory: path to folder in which to save the final images

    Heavily referring to:
    function generateDyanamicImages(params) from
    DynamicImage.m
    from
    http://users.cecs.anu.edu.au/~basura/dynamic_images/code.zip

    Found here:
    https://github.com/hbilen/dynamic-image-nets/blob/master/dicnn/compute_approximate_dynamic_images.m
    Line 7
    '''

    zMean, zMax, zWF, zWR, FirstFrame = getVideoImage(video4d)

    save_images = (
        ('zMean', zMean),
        ('zMax' , zMax ),
        ('zWF' , zWF),
        ('zWR' , zWR),
        ('FirstFrame', FirstFrame)
    )

    for type_, data in save_images:
        if type_ not in KWSET:
            continue

        img_name = image_prefix + '--' + type_ + '.jpg'
        img_path = os.path.join(out_directory, img_name)
        cv2.imwrite(img_path, data)


def getVideoImage(video4d):

    t, h, w, d = video4d.shape

    blue  = np.zeros(shape=(t, h * w), dtype='uint8')
    green = np.zeros(shape=(t, h * w), dtype='uint8')
    red   = np.zeros(shape=(t, h * w), dtype='uint8')

    FirstFrame = None
    if 'FirstFrame' in KWSET:
        FirstFrame = video4d[0]
    blue[:]  = video4d[:, :, :, 0].reshape(t, h * w)
    green[:] = video4d[:, :, :, 1].reshape(t, h * w)
    red[:]   = video4d[:, :, :, 2].reshape(t, h * w)
    
    im_mean_b, im_max_b, im_WF_b, im_WR_b = processVideo(blue , h, w)
    im_mean_g, im_max_g, im_WF_g, im_WR_g = processVideo(green, h, w)
    im_mean_r, im_max_r, im_WF_r, im_WR_r = processVideo(red  , h, w)


    zMean, zMax, zWF, zWR = None, None, None, None
    if 'zMean' in KWSET:
        zMean = getBackRGBImage(im_mean_b, im_mean_g, im_mean_r, h, w)
    if 'zMax' in KWSET:
        zMax  = getBackRGBImage(im_max_b , im_max_g , im_max_r , h, w)
    if 'zWF' in KWSET:
        zWF   = getBackRGBImage(im_WF_b  , im_WF_g  , im_WF_r  , h, w)
    if 'zWR' in KWSET:
        zWR   = getBackRGBImage(im_WR_b  , im_WR_g  , im_WR_r  , h, w)


    return zMean, zMax, zWF, zWR, FirstFrame


def getBackRGBImage(blue, green, red, h, w):
    z = np.zeros(shape=(h, w, 3), dtype='uint8')

    def linearMapping(x):
        minV = x.min()
        maxV = x.max()
        x = x - minV
        x = x / (maxV - minV)
        x = x * 255
        return x

    z[:, :, 0] = linearMapping(blue)
    z[:, :, 1] = linearMapping(green)
    z[:, :, 2] = linearMapping(red)

    return z


def processVideo(color, h, w):

    WF, WR, im_mean, im_max, im_WF, im_WR = None, None, None, None, None, None

    if 'zWF' in KWSET or 'zWR' in KWSET:
        WF, WR = getFowRepresentation(data=color, CVAL=CVAL)

    if 'zMean' in KWSET:
        im_mean = color.mean(axis=0).reshape(h, w)
    if 'zMax' in KWSET:
        im_max = color.max(axis=0).reshape(h, w)
    if 'zWF' in KWSET:
        im_WF = WF.reshape(h, w)
    if 'zWR' in KWSET:
        im_WR = WR.reshape(h, w)

    return im_mean, im_max, im_WF, im_WR


def getFowRepresentation(data, CVAL):

    W_fow, W_rev = None, None
    if 'zWF' in KWSET:
        fdata = data.cumsum(axis=0)
        n, hw = fdata.shape

        for i in range(1, n + 1):
            fdata[i - 1] = fdata[i - 1] / i

        W_fow = liblinearsvr(getNonLinearity(fdata), CVAL)
        del fdata

    if 'zWR' in KWSET:
        # reverse the first dimension
        rdata = np.flip(data, axis=0).cumsum(axis=0)
        n, hw = rdata.shape

        for i in range(1, n + 1):
            rdata[i - 1] = rdata[i - 1] / i

        W_rev = liblinearsvr(getNonLinearity(rdata), CVAL)
        del rdata

    return W_fow, W_rev


def getNonLinearity(data):
    # According to Bilen et al.
    data = np.sign(data) * np.sqrt(np.abs(data)) 

    # Acc to Fernando et al., for CNN
    # data = rootExpandFMap(data)

    return data


def rootExpandFMap(data):
    s = np.sign(data)
    y = np.sqrt(s * data)
    return np.concatenate(y*(s==1), y*(s==-1))


def liblinearsvr(data, C, normD=2):
    if normD != 2:
        raise BaseException('Only normD=2 supported')
    def normalizeL2(X):
        for i in range(len(X)):
            n = norm(X[i])
            if n != 0:
                X[i] /= n
        return X

    data = normalizeL2(data)

    svr = LinearSVR(C=C)
    # labels = np.arange(1, len(data)+1)
    labels = np.arange(0, len(data))

    svr.fit(X=data, y=labels)

    return svr.coef_


def make_dyn_img(folder_with_images, out_directory, image_prefix):
    '''
    Top-level function to be used for generating dynamic images

    folder_with_images: path to folder which contains the images
    out_directory: directory where to save the dynamic image
    image_prefix: the prefix to attach to the image when saving it
    '''
    res = get_image_paths(folder_with_images)
    arr4d = make_4darray(res)

    print(f'Creating for {image_prefix}')
    get_dynamic_image(arr4d, image_prefix, out_directory)

