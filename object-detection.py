import torch
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    return frame

def score_frame(frame):
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def object_detection():
    player = cv2.VideoCapture(0)
    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = player.read()
        results = score_frame(frame)
        frame = plot_boxes(results, frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == 13:
            break
    player.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    object_detection()