import os
import sys
import time

import cv2
import numpy as np

from .dynamic_image import make_dyn_img
from .detect_faces import get_face
from .run_nn_on_dyn_img import get_emotion

def get_cropped_faces(video_path:str, out_directory:str, clean_directory=False):
    '''
    video_path(str): path to video file
    out_directory(str): path to directory to which to save the cropped faces
    clean_directory(bool): deletes all files in the `out_directory`

    Separates a video into frames and crops the face from them.
    NOTE: Assumption, the video has only one prominent face

    '''

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)


    if clean_directory:
        for f in os.listdir(out_directory):
            os.remove(os.path.join(out_directory, f))
        assert len(os.listdir(out_directory)) == 0


    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            # Video has ended
            break

        out_path = os.path.join(out_directory, f'frame{count:05}.jpg')

        if os.path.exists(out_path):
            print(f'Frame {count} already exists! Skipping ...', end='\r')
        else:
            print(f'Saving frame {count}...', end='\r')

            # Send the image object to the func directly, without saving to disk
            get_face(in_image=frame, out_path=out_path)

        count = count + 1
    # Print a new line, to prevent overwriting the 'Saving Frame \d\d\d' message
    print()

    cap.release()
    cv2.destroyAllWindows()

def make_images_uniform_size(image_directory, resize='min'):
    '''
    image_directory: path to directory with cropped faces
    resize: Criterion to resize each image on
        'min' : resize images to min  height, and min  width calculated independently
        'max' : resize images to max  height, and max  width calculated independently
        'mean': resize images to mean height, and mean width calculated independently

    Makes all the images uniform size, based on the selected criterion
    '''

    lh, lw = [], []
    for img in os.listdir(image_directory):
        i = cv2.imread(os.path.join(image_directory, img))
        h, w, _ = i.shape
        lh.append(h)
        lw.append(w)

    lh = np.array(lh)
    lw = np.array(lw)
    
    if resize == 'min':
        print(f'Min Height, Min Width')
        print(f'{lh.min():10d}, {lw.min():9d}')
        res_h, res_w = lh.min(), lw.min()
    elif resize == 'max':
        print(f'Max Height, Max Width')
        print(f'{lh.max():10d}, {lw.max():9d}')
        res_h, res_w = lh.max(), lw.max()
    elif resize == 'mean':
        print(f'Avg Height, Avg Width')
        print(f'{lh.mean():10f}, {lw.mean():9f}')
        res_h, res_w = int(round(lh.mean())), int(round(lw.mean()))
        print(f'{res_h:10}, {res_w:9}')
    else:
        raise ValueError('Option invalid or not implemented')

    # Note: cv2.resize() takes (width, height)
    new_size = (res_w, res_h)

    for ii, img_path in enumerate(os.listdir(image_directory)):
        img_path = os.path.join(image_directory, img_path)
        i = cv2.imread(img_path)
        i = cv2.resize(i, new_size)

        print(f'Saving frame {ii:05}...', end='\r')

        cv2.imwrite(img_path, i)


def get_decision(video_path, folder_frames, folder_dyn_img):
    '''
    Returns a str which is either 'LIE' or 'TRUTH'
    video_path     : path to the video file
    folder_frames  : path to folder which contains the individual frames of the video
    folder_dyn_img : path to folder which contains the dynamic image
    '''

    # This is used to label the image
    image_prefix = os.path.basename(video_path)

    get_cropped_faces(video_path=video_path, out_directory=folder_frames, clean_directory=True)
    make_images_uniform_size(image_directory=folder_frames, resize='min')

    if not os.path.exists(folder_dyn_img):
        os.makedirs(folder_dyn_img)

    make_dyn_img(folder_with_images=folder_frames, out_directory=folder_dyn_img, image_prefix=image_prefix)#'Poppy')


    path_dyn_img = None
    # Find the file associated with current video, by it's image_prefix
    for fname in os.listdir(folder_dyn_img):
        if image_prefix in fname:
            path_dyn_img = fname
            break
    assert len(path_dyn_img) is not None

    path_dyn_img = os.path.join(folder_dyn_img, path_dyn_img)

    emo = get_emotion(dyn_image_path=path_dyn_img)
    print('Predicted emotion:', emo)


    end = time.time()
    print(f'Time taken to predict on a video from dyn img: {end - start}')


    # added on mid sem eval day
    res = 'LIE' if emo != 'others' else 'TRUTH'
    print('*****' * 10)
    print(f'Final Prediction: {res}')
    print('*****' * 10)
    return res

if __name__ == '__main__':
    pass