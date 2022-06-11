import os

import cv2

from src.facial_distance_draw import get_face_landmarks, add_facial_distance_img
from src.qr_code_dist_webcam import disp_qr_code_distance_img_file
from src.qr_code_distance_draw import get_qr_code_bounding_box, get_quadrilateral_skewness


def test_find_qr_code():
    disp_qr_code_distance_img_file(os.path.join('test_data', 'barcode1.png'))
    disp_qr_code_distance_img_file(os.path.join('test_data', 'barcode2.png'))


def test_get_qr_code_bounding_box():
    img2 = cv2.imread(os.path.join('test_data', 'barcode2.png'))
    bbox2 = get_qr_code_bounding_box(img2)
    # assert bbox is not None

    img = cv2.imread(os.path.join('test_data', 'barcode1.png'))
    bbox = get_qr_code_bounding_box(img)
    assert bbox is not None


def test_get_quadrilateral_skewness():
    bbox = [[[285, 168], [226, 267], [375, 245], [431, 116]]]
    skewness = get_quadrilateral_skewness(bbox)
    print(skewness)


def test_get_face_landmark():
    img = cv2.imread(os.path.join('test_data', 'face1.png'))
    get_face_landmarks(img)


def test_add_facial_distance_img():
    img = cv2.imread(os.path.join('test_data', 'face1.png'))
    add_facial_distance_img(img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_add_facial_distance_img()
    # test_get_quadrilateral_skewness()
    # test_find_qr_code()
    # test_get_qr_code_bounding_box()
