import numpy as np

from opencv_dist_measurer.qr_code_distance_algo import LOGI_WEBCAM_SENSOR_MM_HEIGHT, LOGI_WEBCAM_FOCAL_LENGTH_MM

EYE_DISTANCE_MM = 61  # average distance between eyes: 54-68mm


def distance_to_face(landmark, sensor_pixel_height):
    # resource on average facial proportion https://snscourseware.org/snsrcas/files/CW_5c4958cf512fb/facial_feature_examples.pdf

    # average distance between eyes: 54-68mm
    left_eye = np.mean(landmark[0, 37:42], axis=0)
    right_eye = np.mean(landmark[0, 43:48], axis=0)
    eye_pixel_dist = np.linalg.norm(right_eye - left_eye)
    distance_raw = LOGI_WEBCAM_FOCAL_LENGTH_MM * EYE_DISTANCE_MM * sensor_pixel_height / eye_pixel_dist * LOGI_WEBCAM_SENSOR_MM_HEIGHT
    distance_inches = distance_raw * 0.0046
    return distance_inches
