########################## Import
import numpy as np
import imutils
import time
import cv2
import os
import glob
import csv

import warnings
warnings.filterwarnings("ignore")

########################## Control Panel
INPUT=      "input/image/*"
INPUT_VID=  "input/video/img_vid.mp4"
CSV_FILE=	"det_results/detections.csv"
OUTPUT=     "output/output.avi"
YOLO_DIR=   "yolo-coco"
CONFIDENCE=  0.5
THRESHOLD=   0.3
ALLOWED_CLASS=  [2, 5, 7, 9, 11, 12]


########################## Init
# clearing output folder and det_results
print("[INFO] clearing output and det_results folders")
files = glob.glob('output/*')
for f in files:
	os.remove(f)
files = glob.glob('det_results/*')
for f in files:
	os.remove(f)

# reading images
images = sorted(glob.glob(INPUT))
img_num = len(images)
print("[INFO] Loaded {} total images".format(img_num))

# init csv file
print("[INFO] initializing CSV file")
csv_file = open(CSV_FILE, "w+", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_index","detection","left","top","right","bottom"])


# initialize the tracker
from sort import *
tracker = Sort()
memory = {}


# load the COCO class labels our YOLO model was trained on
print("[INFO] Loading names")
labelsPath = os.path.sep.join([YOLO_DIR, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
print("[INFO] Generating Colors")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
print("[INFO] reading weights and cfg")
weightsPath = os.path.sep.join([YOLO_DIR, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_DIR, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


start_time = time.time()
frameIndex = 0
print("[INFO] processing images")

# loop over frames from the video file stream
for image in images:
    start = time.time()
    frame = cv2.imread(image)
    (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []									# class id according to YOLO3 COCO list_of_classes
    												# '0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'.....


    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability)
    		# of the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)				# gets the class of object dedicted
    		confidence = scores[classID]

    		# filter out unwanted classes
    		if classID not in ALLOWED_CLASS:
    			continue

    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > CONFIDENCE:
    			# scale the bounding box coordinates back relative to
    			# the size of the image, keeping in mind that YOLO
    			# actually returns the center (x, y)-coordinates of
    			# the bounding box followed by the boxes' width and
    			# height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")

    			# use the center (x, y)-coordinates to derive the top
    			# and and left corner of the bounding box
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates,
    			# confidences, and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    dets = []										# list of box coordinates and confidence for each object
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
    	for i in idxs.flatten():
    		(x, y) = (boxes[i][0], boxes[i][1])
    		(w, h) = (boxes[i][2], boxes[i][3])
    		dets.append([x, y, x+w, y+h, confidences[i]])


    # reformatting dets (detections) list
    # so that it becomes space-seperated rather than comma-seperated
    # and each number has three digits after the point
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    # updating the tracker with the new detections
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
    	boxes.append([track[0], track[1], track[2], track[3]])
    	indexIDs.append(int(track[4]))
    	memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
    	i = int(0)
    	for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
            	previous_box = previous[indexIDs[i]]
            	(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
            	(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
            	p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
            	p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
            	cv2.line(frame, p0, p1, color, 3)

            csv_writer.writerow([frameIndex , "{}_{}".format(indexIDs[i], LABELS[classIDs[i]]) , x, y, x+w, y+h])

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            # text = "{}".format(indexIDs[i])
            text = "{}: {}".format(indexIDs[i], LABELS[classIDs[i]])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    end = time.time()
    fps = int(1 / (end - start))

    # show fps
    cv2.putText(frame, "FPS: {}".format(str(fps)), (100,100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)

    # saves image file
    cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)


    # increase frame index
    frameIndex += 1
    print("{}/{} frames done".format(frameIndex, img_num))

csv_file.close()
end_time = time.time()
print("[INFO] finished the process in {} seconds".format(end_time - start_time))
