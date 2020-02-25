import os
import pickle

import ffmpeg
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank

PATH_CLASSIFIER = r'./mod_speech/SVC.pkl'
with open(PATH_CLASSIFIER, 'rb') as f:
    MODEL_SVC = pickle.load(f)


def vid_to_aud(path_video, folder_wav):
    assert path_video.endswith('.mp4')
    path_wav = os.path.basename(path_video) + '.wav'

    path_wav =  os.path.join(folder_wav, path_wav)
    if os.path.exists(path_wav):
        os.remove(path_wav)

    ffmpeg.input(path_video).output(path_wav).run(quiet=True)

    return path_wav

def get_mfcc(path_wav):
    (rate,sig) = wavfile.read(path_wav)
    mfcc_feat  = mfcc(sig, rate, nfft=1103)

    new_mfcc = [0 for i in range(len(mfcc_feat[0]))]
    for i in range(len(mfcc_feat[0])):
        s=0
        for j in range(len(mfcc_feat)):
            s+=mfcc_feat[j][i]
        new_mfcc[i]=s/len(mfcc_feat)

    return new_mfcc

def get_decision(mfcc):
    Y_pred = MODEL_SVC.predict([mfcc])
    if(Y_pred==1):
        return 'truth'
    else:
        return 'lie'

def main(filepath):
    path_video = filepath
    folder_wav = r'./wav'

    path_wav = vid_to_aud(path_video, folder_wav)
    mfcc = get_mfcc(path_wav)

    print(f'Wav path: {path_wav}')
    print(f'MFCC: {mfcc}')

    
    result = get_decision(mfcc)
    print(f'{os.path.basename(path_video)} is classified as {result}')


