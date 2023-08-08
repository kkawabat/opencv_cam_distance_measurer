import cv2


def draw_distance_info(dist, coord, img):
    cv2.putText(img, f"{dist:.2f}in", coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def draw_landmarks(landmarks, frame):
    for marks in landmarks:
        cv2.face.drawFacemarks(frame, marks, (255, 255, 255))


def draw_qr_code_bbox(bbox, img_origin):
    # display the image with lines
    # length of bounding box
    n_lines = len(bbox[0])  # Cause bbox = [[[float, float]]], we need to convert float into int and loop over the first element of array
    bbox1 = bbox.astype(int)  # Float to Int conversion

    for i in range(n_lines):
        # draw all lines
        point1 = tuple(bbox1[0, [i][0]])
        point2 = tuple(bbox1[0, [(i + 1) % n_lines][0]])
        cv2.line(img_origin, point1, point2, color=(255, 0, 0), thickness=2)
