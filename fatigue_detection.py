import cv2
from datetime import datetime
from keras.backend import _preprocess_conv2d_input
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import fatigue_cnn as cnn

# load the classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eye_classifier = load_model('./output/model.h5')
lefteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
righteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

eye_label = ['open', 'closed']

fatigue_check_interval = 3.0  # in seconds

prev_time_recording = datetime.now()

left_eye_col = (255, 0, 0)
right_eye_col = (0, 0, 255)


closed_last_interval = False

min_observations_count = 5

open_observations_count = 0  # used to record number of times eyes were detected as being open during fatigue_check_interval
closed_observations_count = 0  # used to record number of times eyes were detected as being closed during fatigue_check_interval

# value of open_closed_ratio at or above which subject is believed to be showing signs of fatigue during last
# fatigue_check_interval - this is to counteract possible false neg/pos observations from model
fatigue_warning_threshold = 0.7


def check_fatigue():
    global closed_observations_count
    global open_observations_count
    global closed_last_interval

    if closed_observations_count + open_observations_count > min_observations_count:
        # scale of 0 - 1 where 0 means eyes fully open for fatigue_check_interval duration and 1 means eyes fully
        # closed for fatigue_check_interval duration
        open_closed_ratio = closed_observations_count / (closed_observations_count + open_observations_count)

        # reset counters
        closed_observations_count = 0
        open_observations_count = 0

        met_half_threshold = (open_closed_ratio >= fatigue_warning_threshold / 2)
        met_threshold = (open_closed_ratio >= fatigue_warning_threshold) or \
                        (met_half_threshold and closed_last_interval)

        if met_threshold:
            print('CLOSED FOR TOO LONG!')

        closed_last_interval = met_half_threshold


def handle_eye_status(eye, roi_side, label_text="", flip=False, offset=-10, col=right_eye_col):
    global closed_observations_count
    global open_observations_count
    for (ex, ey, ew, eh) in eye:
        # If eye ROI is on wrong side of face, disregard it
        if flip:
            if ex < w / 2:
                continue
        else:
            if ex > w / 2:
                continue
        cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), col, 2)
        roi_side = roi_side[ey:ey + eh, ex:ex + ew]
        try:
            roi_side = cv2.resize(roi_side, (cnn.img_size, cnn.img_size), interpolation=cv2.INTER_AREA)
        except:
            continue

        if np.sum([roi_side]) != 0:
            roi = roi_side.astype('float') / 255.0
            if flip:
                roi = cv2.flip(roi, 1)
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = eye_classifier.predict(roi)[0][0]
            eye_status = eye_label[prediction < 0.5]
            label = label_text + " " + eye_status + ' ' + str(prediction)

            if eye_status == 'open':
                open_observations_count += 1
            else:
                closed_observations_count += 1

            label_position = (x, y + offset)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)


# get webcam capture and set size
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set width of window
cap.set(4, 480)  # set height of window


# check if webcam opened
if not cap.isOpened():
    raise IOError('Webcam did not open correctly.\nAborting...')


# open the windows with webcam feed
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # make colour scale gray

    # detect faces
    faces = face_cascade.detectMultiScale(
        gray,  # inputs the frame as grayscale
        scaleFactor=1.05,  # specifies how much the image size is reduced at each image scale
        minNeighbors=7,
        # specifies how many neighbours each rectangle should have. Increase this to reduce false positives
        minSize=(30, 30)  # minimum rectangle size to be considered a face
    )
    # mark the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray_left = gray[y:y + h, x:x + w]
        roi_gray_right = gray[y:y + h, x:x + w]
        roi_colour = frame[y:y + h, x:x + w]

        left_eye = lefteye_cascade.detectMultiScale(
            roi_gray_left,  # inputs the frame within the face rectangle as grayscale
            scaleFactor=1.05,  # specifies how much the image size is reduced at each image scale
            minNeighbors=15,
            # specifies how many neighbours each rectangle should have. Increase this to reduce false positives
            minSize=(5, 5)  # minimum rectangle size to be considered eyes
        )

        right_eye = righteye_cascade.detectMultiScale(
            roi_gray_right,  # inputs the frame within the face rectangle as grayscale
            scaleFactor=1.05,  # specifies how much the image size is reduced at each image scale
            minNeighbors=15,
            # specifies how many neighbours each rectangle should have. Increase this to reduce false positives
            minSize=(5, 5)  # minimum rectangle size to be considered eyes
        )

        handle_eye_status(left_eye, roi_gray_left, "Left Eye:", True, -40, left_eye_col)
        handle_eye_status(right_eye, roi_gray_right, "Right Eye:", False, -10, right_eye_col)

    cur_time = datetime.now()
    time_diff = (cur_time - prev_time_recording).total_seconds()
    if time_diff >= fatigue_check_interval:
        check_fatigue()
        prev_time_recording = cur_time

        # show the frame
    cv2.imshow('Eyes', frame)  # open window

    # press 'ESC' key to terminate the program
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# cleans up
cap.release()
cv2.destroyAllWindows()
