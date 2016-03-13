import cv2
import sys
import numpy as np
import scipy as sp
from neural_network import *
from data_preprocess import random_crop
from data_preprocess import normalize_single_image

foldername = '../TrainedModels/'
filename = 'fer1_best_model'
prediction_fn = load_and_build_model(foldername+filename)


# adding random cropping into the prediction. 
foldername1 = '../TrainedModelsBatch1/'
filename1= 'fer2_aug_cf_best_model'
prediction_fn_crop = load_and_build_model(foldername1+filename1, mode = 'probability')


#cascPath = sys.argv[1]
cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_rows = 144 * 2
video_cols = 256 * 2
video_capture.set(3, video_cols)
video_capture.set(4, video_rows)

status = np.zeros((video_rows, video_cols, 3))


fig_path = '../figs/'
# emotions = ['Happy', 'Sad', 'Neutral',
#             'Disgust', 'Fear', 'Angry', 'Surprise']
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

emotion_dict_5 = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Neutral'}

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
img_shape = (230, 220, 3)


def get_emotion_face(emotion):
    img = cv2.imread(fig_path + emotion + '.png')
    img = cv2.resize(img, (50, 50))
    return img

np.random.seed()

iscrop = False

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
        my_face = frame[y:y+h, x:x+w]
        my_face = cv2.cvtColor(my_face, cv2.COLOR_BGR2GRAY)
        #my_face = my_face/255. #this is not normalize with respect to lighting
        my_face = my_face.astype(float)/255.
        # normalize and reshape my_face
        input_face = sp.misc.imresize(my_face, (48, 48))

        if iscrop:
            left_shift = np.random.randint(low = -3, high = 4, size = 3)
            right_shift = np.random.randint(low = -3, high = 4, size = 3)
            input_crop = np.zeros((3,1,42,42))
            for i in range(3):
                input_crop[i,0,:,:] = random_crop(img = input_face, crop_sz = 42, left_shift = left_shift[i], right_shift = right_shift[i])
            y_pred = np.argmax(np.mean(prediction_fn_crop(input_crop),axis = 0))
        else:
            input_face = input_face.reshape(1, 1, 48, 48)
            input_face = normalize_single_image(input_face)
            y_pred = prediction_fn(input_face)
            y_pred = y_pred[0]
        # get the emotion image
        # cv2.imshow('image', my_face)
        img = get_emotion_face(emotion_dict[y_pred])
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
