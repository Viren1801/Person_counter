import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def non_max_suppression_fast(boxes, overlapthresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapthresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


# Recording result
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('Output.avi', fourcc, 15.0, (800, 600))

# Loading file
cap = cv2.VideoCapture("test.mp4")
# cap = cv2.VideoCapture(0)


# Initial system
tracker = CentroidTracker(maxDisappeared=20, maxDistance=90)
fps_start_time = datetime.datetime.now()
total_frames = 0
fps = 0
live_persons = 0
total_persons = 0
flag = 0
person_centroids = []

while cap.isOpened():
    ret, frame = cap.read()

    # Resizing working window
    frame = imutils.resize(frame, width=600)

    # call FPS system
    total_frames = total_frames + 1

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    person_detections = detector.forward()
    rects = []
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")
            rects.append(person_box)

    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)

    objects = tracker.update(rects)
    for (person_id, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        text = "ID: {}".format(person_id)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        # Alert system
        try:
            if person_centroids[person_id]:
                pass
        except IndexError:
            person_centroids.append([])
        centroid = (x2 - x1, y2 - y1)
        if flag % 6 == 0:
            person_centroids[person_id].append(centroid)
            if len(person_centroids[person_id]) >= 30:
                person_centroids[person_id].pop(0)

        x_avg, y_avg = 0, 0
        for object_ in person_centroids[person_id]:
            x_avg += object_[0]
            y_avg += object_[1]

        x_avg /= len(person_centroids[person_id])
        y_avg /= len(person_centroids[person_id])

        if abs(x_avg - centroid[0]) < 10 and abs(y_avg - centroid[1]) < 10:
            print("Alert", person_id)
        else:
            print("Clear", person_id)

    # FPS counting system
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

    # Person counting system
    live_persons = len(objects)
    total_persons = len(person_centroids)

    live = "Live: {}".format(live_persons)
    total = "Total: {}".format(total_persons)

    cv2.putText(frame, live, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    cv2.putText(frame, total, (5, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

    # Resizing the output
    resize = cv2.resize(frame, (800, 600))
    cv2.imshow("Detecting", resize)
    video_out.write(resize)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
