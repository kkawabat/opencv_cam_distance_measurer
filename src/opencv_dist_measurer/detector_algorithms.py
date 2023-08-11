from os.path import dirname, join, abspath

import cv2
import numpy as np

MODEL_DIR = join(dirname(abspath(__file__)), 'model')

FACE_DETECTOR = cv2.CascadeClassifier(join(MODEL_DIR, 'lbpcascade_frontalface_improved.xml'))

FACE_LANDMARK_DETECTOR = cv2.face.createFacemarkLBF()
FACE_LANDMARK_DETECTOR.loadModel(join(MODEL_DIR, 'lbfmodel.yaml'))
QR_DETECTOR = cv2.QRCodeDetector()


def get_face_landmarks(img):
    faces = FACE_DETECTOR.detectMultiScale(img,
                                           scaleFactor=1.3,
                                           minNeighbors=4,
                                           minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE
                                           )
    if len(faces) == 0:
        return []
    _, landmarks = FACE_LANDMARK_DETECTOR.fit(img, faces=faces)
    return landmarks


def get_qr_code_bounding_box(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bbox, _ = QR_DETECTOR.detectAndDecode(gray_img)
    if bbox is None:
        return []
    if not is_valid_bbox(bbox):
        return []
    return bbox


def is_valid_bbox(bbox):
    skewness = get_quadrilateral_skewness(bbox)
    if skewness > .5:
        return False
    return True


def get_quadrilateral_skewness(bbox):
    # skewness calc 'https://www.engmorph.com/skewness-finite-elemnt'
    ll, ul, ur, lr = bbox[0]
    a = get_angle(ul, ur, lr)
    b = get_angle(ur, lr, ll)
    c = get_angle(lr, ll, ul)
    d = get_angle(ll, ul, ur)
    angle_max = max([a, b, c, d])
    angle_min = min([a, b, c, d])
    return max([(angle_max - 90) / 90, (90 - angle_min) / 90])


def get_angle(a, b, c):
    ba = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([b[0] - c[0], b[1] - c[1]])
    dot = np.dot(ba, bc)
    angle = np.degrees(np.abs(np.arccos(dot / (np.linalg.norm(ba) * np.linalg.norm(bc)))))
    return angle
