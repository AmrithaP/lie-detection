
import numpy as np
import argparse
import cv2

from constants import (
    PATH_PROTOTXT,
    PATH_CAFFEMODEL,
    CAFFE_CONF
)

# Adapted from:
# https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/blob/master/detect_faces.py

# load our serialized model from disk
print("[INFO] loading caffemodel...")
net = cv2.dnn.readNetFromCaffe(PATH_PROTOTXT, PATH_CAFFEMODEL)

def get_face(in_image, out_path):
    '''
    in_image: path for input image (str) OR
              image object (ndarray(dtype=uint8)) (height, width, depth)
    out_path: path to save output image
    '''

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it

    if type(in_image) == np.ndarray and len(in_image.shape) == 3:
        # if input `in_image` is an np.ndarray, then use it directly
        image = in_image

    elif type(in_image) == str:
        # if input `in_image` is a str i.e. a path, use imread to read it
        image = cv2.imread(in_image)
    else:

        raise ValueError('`in_image` is not a path or an image object (numpy.ndarray)'
                         f'It is actually, {type(in_image)}'
            )

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
           )

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        calc_confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if calc_confidence > CAFFE_CONF:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
     
            extract_img = image[startY:endY+1, startX:endX+1]

    # save the cropped face image
    cv2.imwrite(out_path, extract_img)
