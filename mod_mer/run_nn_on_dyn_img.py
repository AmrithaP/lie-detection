
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from constants import PATH_MER_MODEL

print('tf version:', tf.__version__)

# Better to use 1.15.0, as 1.15.0 was used to create the trained model on Colab
assert tf.__version__ == '1.15.0'

# Global Constants being used
IMAGE_DIM = (112, 112)
IMAGE_DEPTH = 1
lmodel = keras.models.load_model(PATH_MER_MODEL)

print(f'IMAGE_DIM   = {IMAGE_DIM}')
print(f'IMAGE_DEPTH = {IMAGE_DEPTH}')

emo_list = ['repression', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'others']
emo_dict = {'repression': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'sadness': 4, 'surprise': 5, 'others': 6}


def get_emotion(dyn_image_path):

    # Path to dyn img
    test_image = dyn_image_path

    timg = cv2.imread  (test_image)
    print('*****' * 10)
    print('read image')
    print('*****' * 10)
    print()
    
    timg = cv2.resize  (timg, (*IMAGE_DIM,))
    print('*****' * 10)
    print('resize')
    print('*****' * 10)
    print()
    
    timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    print('*****' * 10)
    print('grayscale convert')
    print('*****' * 10)
    print()

    timg = timg.reshape((*timg.shape, 1))
    print('*****' * 10)
    print('add gray dim')
    print('*****' * 10)
    print()

    arr_timg = np.expand_dims(timg, axis=0)

    arr_pred = lmodel.predict(x=arr_timg)
    pred0 = arr_pred[0]

    table = []
    for emo, emon in emo_dict.items():
        table.append((emo, emon, pred0[emon]))
    table.sort(key=lambda x: x[2], reverse=True)

    for emo, emon, num in table:
        print(f'{emo:10s} ({emon:1d}) :  {num:}')

    final_pred = table[0][0]
    print('\n\nFinal Prediction:')
    print(final_pred)

    return final_pred
