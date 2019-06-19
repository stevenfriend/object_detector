from utils import *
import numpy as np
import cv2
import sys

# Load the caffe model
model_name = 'MobileNetSSD_deploy.caffemodel'
model_proto = 'MobileNetSSD_deploy.prototxt'

net = cv2.dnn.readNetFromCaffe(model_proto, model_name)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_objects_and_draw_boxes(net, frame)
    cv2.imshow("Object Detector", cv2.resize(frame, (1000, 700)))

    k = cv2.waitKey(10)
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
