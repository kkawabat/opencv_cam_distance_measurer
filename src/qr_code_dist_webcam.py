"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2

from src.qr_code_distance_draw import add_qr_code_distance_img


def display_qr_dist_webcam():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret_val, img = cam.read()
        img = add_qr_code_distance_img(img)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def disp_qr_code_distance_img_file(file_path):
    # read the QRCODE image
    # in case if QR code is not black/white it is better to convert it into grayscale
    img_origin = cv2.imread(file_path)
    img = add_qr_code_distance_img(img_origin)
    # display the result
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    display_qr_dist_webcam()


if __name__ == '__main__':
    main()
