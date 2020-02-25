from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
import ffmpeg 

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
 


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")

# path_shape_predictor = r"C:\Users\HP\Desktop\GUI\LIE_DETECTION\shape_predictor_68_face_landmarks.dat"
path_shape_predictor = r'./mod_blink/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_shape_predictor)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def get_ear_from_video(video_path, analyse_time=15):
    '''
    video_path: path to video file
    analyse_time: Time in seconds to analyse the video
    '''

    vs = FileVideoStream(video_path).start()
    #vs = VideoStream(src=0).start()
    fileStream = True


    li=[]

    # loop over frames from the video stream
    t_end = time.time() + analyse_time

    while time.time() < t_end:
        
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        frame = vs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            li.append(ear)       

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SIT ERECT AND FACE CAMERA", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
     
        # show the frame

        # if the `q` key was pressed, break from the loop
        cv2.imshow("Test time", frame)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    li = sorted(li)
    l = len(li)

    if l%2 == 0:
        thres = (li[l//2] + li[l//2+1])/2
    else:
        thres = li[l//2]
    # print(f'li:\n', li, sep='')

    # EAR = thres
    print(f'EAR = {thres}')
    return thres



def get_blinks(video_path, ear_val):
    '''
    video_path: path to video file
    ear_val: float value for eye aspect ratio

    returns
    total blinks in the entire video
    '''
    thres = ear_val
    EYE_AR_THRESH = thres
    EYE_AR_CONSEC_FRAMES = 3

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0


    vs1 = FileVideoStream(video_path).start()
    fileStream = True
    # time.sleep(2.0)

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs1.more():
            break

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs1.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # print(f'left : {leftEye}')
            # print(f'right: {rightEye}')
            # if (leftEye[1][1] - leftEye[5][1]) < 1:
            #     print(f'{time.time()} Blink Left')

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # if the `q` key was pressed, break from the loop
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    # vs.stop()

    return TOTAL # no of blinks


def determine_truth_lie(blinks, BlinksPerMinute=26):
    '''
    Returns string of truth or lie
    '''

    print("BLINKS : ", blinks)
    if abs(blinks- BlinksPerMinute) > 7:
        # print("lie")
        return "lie"
    else:
        # print("truth")
        return "truth"


def main(filepath):
    # args = dict()
    "blink_detection_demo.mp4"
    "ip_split.avi"
    "ip.avi"

    #path = 'ip_split.avi'
    path = filepath #r"C:\Users\Mandar\Desktop\lie\full truth.mp4"
    #path = 'Blinking.mp4'
    dur = float(ffmpeg.probe(path)["streams"][0]["duration"])
    print("DURATION:",dur)
    ear = get_ear_from_video(path)-0.03

    blinks = get_blinks(path, ear_val=ear)
    print("Total blinks:",blinks)
    blinks = int((60*blinks)/dur)

    print('Result:')
    res = determine_truth_lie(blinks=blinks)
    print(res)
    return res
