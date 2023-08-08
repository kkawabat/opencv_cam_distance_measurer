import cv2

from src.opencv_dist_measurer.distance_draw import add_qr_code_distance_img, add_facial_distance_img


def dist_webcam(detector='face'):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret_val, img = cam.read()

        if detector == 'face':
            img = add_facial_distance_img(img)
        elif detector == 'qr':
            img = add_qr_code_distance_img(img)

        cv2.imshow("img", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def dist_image_file(file_path, detector='face'):
    # read the QRCODE image
    # in case if QR code is not black/white it is better to convert it into grayscale
    img = cv2.imread(file_path)

    if detector == 'face':
        img = add_facial_distance_img(img)
    elif detector == 'qr':
        img = add_qr_code_distance_img(img)

    # display the result
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dist_webcam()
