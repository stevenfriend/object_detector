import cv2
import numpy as np

# The categories the model was trained for
categories = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motobike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def detect_objects_and_draw_boxes(net, image):

    h, w = image.shape[:2]

    resized = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)

    # Feed the blob as input to our deep learning model
    net.setInput(blob)

    # Run detection
    detections = net.forward()[0][0]

    # Loop over the detections
    for i in range(len(detections)):

        # Each detection is of the following format
        # [0, predicted_category, confidence_value, x1, y1, x2, y2]

        confidence = round(detections[i][2]*100, 2)
        category_index = int(detections[i][1])

        # Confidence threshold
        if confidence > 60:

            # Scale up the box coordinates
            box = detections[i][3:] * np.array([w, h, w, h])

            # Convert them to input
            (x1, y1, x2, y2) = box.astype("int")

            object_name = categories[category_index]
            display = object_name + ":" + str(confidence) + "%"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, display, (x1, y1), font, 1, (0, 255, 0), 2)
