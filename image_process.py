# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os
import glob

files = glob.glob('output/*')
for f in files:
	os.remove(f)

files = glob.glob('det_results/*')
for f in files:
	os.remove(f)

INPUT=      	"input/image/*"
OUTPUT=     	"output/"
YOLO_DIR=   	"yolo-coco"
CONFIDENCE= 	0.6
THRESHOLD=   	0.3
ALLOWED_CLASS=  [2, 5, 7, 9, 11, 12]

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

# load images
images = glob.glob(INPUT)
print("[INFO] {} total images".format(len(images)))
image_counter = 0

start_time = time.time()

# loop over frames from the video file stream
for image in images:

	image_name_no_ext = image.split('/')[-1].split('.')[0]
	text_file = open("det_results/{}.txt".format(image_name_no_ext), 'w+')


	img = cv2.imread(image)

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)

	layerOutputs = net.forward(ln)


	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []									# class id according to YOLO3 COCO list_of_classes
													# '0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'.....
	(H, W) = img.shape[:2]

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

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
		    # extract the bounding box coordinates
		    (x, y) = (int(box[0]), int(box[1]))
		    (w, h) = (int(box[2]), int(box[3]))

		    # draw a bounding box rectangle and label on the image
		    color = [int(c) for c in COLORS[i % len(COLORS)]]
		    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


		    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		    # text = "{}".format(indexIDs[i])
		    text = "{}".format( LABELS[classIDs[i]])
		    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		    text_file.write("{} {} {} {} {} {}\n".format(text, confidences[i], x, y, x+w, y+h))

		    i += 1


	# saves image file
	cv2.imwrite("{}/{}".format(OUTPUT, image.split("/")[2]), img)

	text_file.close()
	image_counter += 1
	print("{}/{} {}  Done".format(image_counter, len(images), image_name_no_ext))

end_time = time.time()
print("[INFO] finished the process in {} seconds".format(end_time - start_time))
