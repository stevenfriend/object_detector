from utils import *
import numpy as np
import cv2
import sys

data_file = sys.argv[1]
file_type = None

# Load the caffe model
model_name = 'MobileNetSSD_deploy.caffemodel'
model_proto = 'MobileNetSSD_deploy.prototxt'

net = cv2.dnn.readNetFromCaffe(model_proto, model_name)

img = cv2.imread(data_file)
detect_objects_and_draw_boxes(net, img)

cv2.imshow("Object Detector", img)
cv2.waitKey(0)
