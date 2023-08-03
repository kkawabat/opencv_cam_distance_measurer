import os

import cv2
import numpy as np

from opencv_dist_measurer.face_dist_algo import distance_to_face

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# source "https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e"
FACE_DETECTOR = cv2.CascadeClassifier(os.path.join(MODEL_DIR, 'haarcascade_face_model.xml'))

FACE_LANDMARK_DETECTOR = cv2.face.createFacemarkLBF()
FACE_LANDMARK_DETECTOR.loadModel(os.path.join(MODEL_DIR, 'face_landmark_lbfmodel.yaml'))


def add_facial_distance_img(img):
    landmarks = get_face_landmarks(img)
    img = draw_distance_to_face(landmarks, img)
    img = draw_landmarks(landmarks, img)
    return img


def get_face_landmarks(img):
    try:
        faces = FACE_DETECTOR.detectMultiScale(img)
        # see landmark indices https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        _, landmarks = FACE_LANDMARK_DETECTOR.fit(img, faces)
    except:
        return None
    return landmarks


def draw_distance_to_face(landmarks, img):
    if landmarks is not None:
        try:
            dist = distance_to_face(landmarks[0], len(img))
            cv2.putText(img, f"{dist:.2f}in", tuple(landmarks[0][0, 20].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except:
            return img
    return img


def draw_landmarks(landmarks, img):
    try:
        validate_facial_landmarks(landmarks)
    except Exception as e:
        return img

    for landmark in landmarks:
        for x, y in landmark[0]:
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 255), 1)
    return img


def validate_facial_landmarks(landmarks):
    if landmarks is None:
        raise Exception("Not all facial landmarks detected")
