import os

import cv2

from src.opencv_dist_measurer.detector_algorithms import get_face_landmarks, get_quadrilateral_skewness, get_qr_code_bounding_box
from src.opencv_dist_measurer.dist_main import dist_image_file
from src.opencv_dist_measurer.distance_draw import add_facial_distance_img


def test_find_qr_code():
    dist_image_file(os.path.join('test_data', 'barcode1.png'), 'qr')
    dist_image_file(os.path.join('test_data', 'barcode2.png'), 'qr')


def test_get_qr_code_bounding_box():
    img2 = cv2.imread(os.path.join('test_data', 'barcode2.png'))
    bbox2 = get_qr_code_bounding_box(img2)
    assert bbox2 is not None

    img = cv2.imread(os.path.join('test_data', 'barcode1.png'))
    bbox = get_qr_code_bounding_box(img)
    assert bbox is not None

    img = cv2.imread(os.path.join('test_data', 'face1.png'))
    bbox = get_qr_code_bounding_box(img)
    assert bbox is None


def test_get_quadrilateral_skewness():
    bbox = [[[285, 168], [226, 267], [375, 245], [431, 116]]]
    skewness = get_quadrilateral_skewness(bbox)
    print(skewness)


def test_get_face_landmark():
    # img = cv2.imread(os.path.join('test_data', 'face1.png'))
    img = cv2.imread(os.path.join('test_data', 'lena.jpg'))
    get_face_landmarks(img)
    img_no_face = img[0:240, 0:320, :]
    get_face_landmarks(img_no_face)


def test_add_facial_distance_img():
    img = cv2.imread(os.path.join('test_data', 'face1.png'))
    # img = cv2.imread(os.path.join('test_data', 'lena.jpg'))
    add_facial_distance_img(img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_no_face = img[0:240, 0:320, :]
    add_facial_distance_img(img_no_face)
    cv2.imshow("img", img_no_face)


if __name__ == '__main__':
    test_add_facial_distance_img()
    # test_get_quadrilateral_skewness()
    # test_find_qr_code()
    # test_get_qr_code_bounding_box()
