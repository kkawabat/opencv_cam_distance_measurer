import cv2
import numpy as np

from src.qr_code_distance_algo import dist_to_qr_code, draw_distance_to_qr_code

detector = cv2.QRCodeDetector()


def add_qr_code_distance_img(img):
    bbox = get_qr_code_bounding_box(img)
    img = draw_distance_to_qr_code(bbox, img)
    img = draw_qr_code_bbox(bbox, img)
    return img


def get_qr_code_bounding_box(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data, bbox, straight_qrcode = detector.detectAndDecode(gray_img)
    return bbox


def draw_qr_code_bbox(bbox, img_origin):
    if bbox is not None:
        try:
            # display the image with lines
            # length of bounding box
            n_lines = len(bbox[0])  # Cause bbox = [[[float, float]]], we need to convert float into int and loop over the first element of array
            bbox1 = bbox.astype(int)  # Float to Int conversion

            validate_bbox(bbox)

            for i in range(n_lines):
                # draw all lines
                point1 = tuple(bbox1[0, [i][0]])
                point2 = tuple(bbox1[0, [(i + 1) % n_lines][0]])
                cv2.line(img_origin, point1, point2, color=(255, 0, 0), thickness=2)
        except:
            return img_origin
    return img_origin


def validate_bbox(bbox):
    skewness = get_quadrilateral_skewness(bbox)
    if skewness > .5:
        raise Exception(f'invalid qr bound')


def get_quadrilateral_skewness(bbox):
    # skewness calc 'https://www.engmorph.com/skewness-finite-elemnt'
    ll, ul, ur, lr = bbox[0]
    a = get_angle(ul, ur, lr)
    b = get_angle(ur, lr, ll)
    c = get_angle(lr, ll, ul)
    d = get_angle(ll, ul, ur)
    angle_max = max([a, b, c, d])
    angle_min = min([a, b, c, d])
    return max([(angle_max - 90)/90, (90 - angle_min)/90])


def get_angle(a, b, c):
    ba = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([b[0] - c[0], b[1] - c[1]])
    dot = np.dot(ba, bc)
    angle = np.degrees(np.abs(np.arccos(dot / (np.linalg.norm(ba) * np.linalg.norm(bc)))))
    return angle
