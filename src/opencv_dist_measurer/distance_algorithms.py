import numpy as np

#  face params
EYE_DISTANCE_MM = 61  # average distance between eyes: 54-68mm

#  qr code params
LOGI_WEBCAM_FOCAL_LENGTH_MM = 3.67  # focal length for C920HD webcam in mm
LOGI_WEBCAM_SENSOR_MM_HEIGHT = 2.72  # for C920HD webcam in mm derived here 'https://stackoverflow.com/a/50649184/4231985'
QR_CODE_SIZE_MM = 37  # QR CODE SIZE in mm


def distance_to_face(landmark, sensor_pixel_height):
    # resource on average facial proportion https://snscourseware.org/snsrcas/files/CW_5c4958cf512fb/facial_feature_examples.pdf
    left_eye = np.mean(landmark[0, 37:42], axis=0)
    right_eye = np.mean(landmark[0, 43:48], axis=0)
    eye_pixel_dist = np.linalg.norm(right_eye - left_eye)
    distance_raw = LOGI_WEBCAM_FOCAL_LENGTH_MM * EYE_DISTANCE_MM * sensor_pixel_height / eye_pixel_dist * LOGI_WEBCAM_SENSOR_MM_HEIGHT
    distance_inches = distance_raw * 0.0046
    return distance_inches


def distance_to_qr(bbox, sensor_pixel_height):
    # focal_length_calc 'https://stackoverflow.com/a/50649184/4231985'
    # reading material https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    qr_code_pixel_height = _get_height(bbox)
    distance_raw = LOGI_WEBCAM_FOCAL_LENGTH_MM * QR_CODE_SIZE_MM * sensor_pixel_height / qr_code_pixel_height * LOGI_WEBCAM_SENSOR_MM_HEIGHT

    distance_inches = distance_raw * 0.0046
    return distance_inches


def _get_height(bbox):
    height = bbox[0][2][1] - bbox[0][0][1]
    return height
