import cv2

LOGI_WEBCAM_FOCAL_LENGTH_MM = 3.67  # focal length for C920HD webcam in mm
LOGI_WEBCAM_SENSOR_MM_HEIGHT = 2.72  # for C920HD webcam in mm derived here 'https://stackoverflow.com/a/50649184/4231985'
QR_CODE_SIZE_MM = 37  # QR CODE SIZE in mm


def get_height(bbox):
    height = bbox[0][2][1] - bbox[0][0][1]
    return height


def draw_distance_to_qr_code(bbox, img):
    try:
        dist = dist_to_qr_code(bbox, len(img))
        cv2.putText(img, f"{dist:.2f}in", tuple(bbox[0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    except:
        return img
    return img


def dist_to_qr_code(bbox, sensor_pixel_height):
    # focal_length_calc 'https://stackoverflow.com/a/50649184/4231985'
    # reading material https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    qr_code_pixel_height = get_height(bbox)
    distance_raw = LOGI_WEBCAM_FOCAL_LENGTH_MM * QR_CODE_SIZE_MM * sensor_pixel_height / qr_code_pixel_height * LOGI_WEBCAM_SENSOR_MM_HEIGHT

    distance_inches = distance_raw * 0.0046
    return distance_inches
