from .detector_algorithms import get_qr_code_bounding_box, get_face_landmarks
from .distance_algorithms import distance_to_qr, distance_to_face
from .drawer import draw_distance_info, draw_qr_code_bbox, draw_landmarks


def add_qr_code_distance_img(frame):
    bbox = get_qr_code_bounding_box(frame)
    if len(bbox) == 0:
        return

    coord = tuple(bbox[0][0].astype(int))
    dist = distance_to_qr(bbox, len(frame))

    draw_distance_info(dist, coord, frame)
    draw_qr_code_bbox(bbox, frame)


def add_facial_distance_img(frame):
    landmarks = get_face_landmarks(frame)
    if len(landmarks) == 0:
        return frame

    dist = distance_to_face(landmarks[0], len(frame))
    coord = tuple(landmarks[0][0, 20].astype(int))
    draw_distance_info(dist, coord, frame)
    draw_landmarks(landmarks, frame)
