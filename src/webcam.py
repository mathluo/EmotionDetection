import cv2
import sys
import numpy as np
import scipy as sp
from neural_network import *

foldername = '../TrainedModels/'
filename = 'fer1_best_model'

prediction_fn = load_and_build_model(foldername+filename)


cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_rows = 144 * 2
video_cols = 256 * 2
video_capture.set(3, video_cols)
video_capture.set(4, video_rows)

status = np.zeros((video_rows, video_cols, 3))


fig_path = '../figs/'
emotions = ['Happy', 'Sad', 'Neutral',
            'Disgust', 'Fear', 'Angry', 'Surprise']
emotion_dict = {0: 'Happy', 1: 'Sad', 2: 'Neutral',
                3: 'Disgust', 4: 'Fear', 5: 'Angry', 6: 'Surprise'}

img_shape = (230, 220, 3)


def get_emotion_face(emotion):
    img = cv2.imread(fig_path + emotion + '.png')
    img = cv2.resize(img, (50, 50))
    return img

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # extract the face
        my_face = frame[y:y+h, x:x+w, :]/255.
        # normalize and reshape my_face
        input_face = sp.misc.imresize(my_face[:, :, 0], (48, 48))
        input_face = input_face.reshape(1, 1, 48, 48)

        # classify the face to emotions
        y_pred = prediction_fn(input_face)
        print y_pred[0]
        # get the emotion image
        # cv2.imshow('image', my_face)
        img = get_emotion_face(emotion_dict[y_pred[0]])
        # set the emotion image to the frame
        frame[y:y+img.shape[0], x:x+img.shape[1], :] = img

    # frame = np.hstack((frame, status))
    cv2.imshow('image', frame)

    # cv2.imshow(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
