import cv2

from opencv_dist_measurer.facial_distance_draw import add_facial_distance_img


def display_face_dist_webcam():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret_val, img = cam.read()
        img = add_facial_distance_img(img)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_face_dist_webcam()
