# import the necessary packages
import numpy as np
# import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
from scipy import spatial
import cv2
from input_retrieval import *
# import pafy
# from threading import Thread
# import sys
# from queue import Queue
import multiprocessing

from atcs import traffic_control

#All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
DEFAULT_MPP = 0.05  # meters per pixel fallback when calibration not provided (kept for potential future use)

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU, inputVideoPathList, outputVideoPathAll, meters_per_pixel, distance_km, speed_limit_kmh, display_advice, max_controller_seconds, max_controller_cycles = parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles 
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count,lane, avg_speed_text=None, recommendation_text=None):
	cv2.putText(
		frame, #Image
		'Detected Vehicles in Lane' +str(lane+1)+"::  =  "  + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.75, #Size
		(0, 255, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)
	y = 45
	if avg_speed_text:
		cv2.putText(frame, avg_speed_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		y += 25
	# recommendation_text removed as we no longer show speed advice

# PURPOSE: Determining if the box-mid point cross the line or are within the range of 5 units
# from the line
# PARAMETERS: X Mid-Point of the box, Y mid-point of the box, Coordinates of the line 
# RETURN: 
# - True if the midpoint of the box overlaps with the line within a threshold of 5 units 
# - False if the midpoint of the box lies outside the line and threshold
def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
	x1_line, y1_line, x2_line, y2_line = line_coordinates #Unpacking

	if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
		(y_mid_point >= y1_line and y_mid_point <= y2_line+5):
		return True
	return False

# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		#os.system('clear') # Equivalent of CTRL+L on the terminal
		# print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#Draw a green dot in the middle of the box
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream,outputVideoPath):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
#			the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
#		  False if the box was not present in the previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
	current_detections = {}
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in np.array(idxs).flatten():
			# extract the bounding box coordinates
			if i >= len(boxes) or i >= len(classIDs):
				continue
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			if w <= 0 or h <= 0:
				continue
			
			centerX = int(x + (w//2))
			centerY = int(y + (h//2))

			# When the detection is in the list of vehicles, AND
			# it crosses the line AND
			# the ID of the detection is not present in the vehicles
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count 
				#
				if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections):
					vehicle_count += 1
					# vehicle_crossed_line_flag += True
				# else: #ID assigning
					#Add the current detection mid-point of box to the list of detected items
				# Get the ID corresponding to the current detection

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close, 
				# then assign a new ID to current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 

				#Display the ID at the center of the box
				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, current_detections

def compute_and_update_speed(lane, current_detections, prev_positions_by_id, last_seen_time_by_id, mpp, vehicle_count_instance):
	now = time.time()
	inst_speeds_mps = []
	for (cx, cy), ID in current_detections.items():
		prev_pt = prev_positions_by_id.get(ID)
		prev_t = last_seen_time_by_id.get(ID)
		if prev_pt is not None and prev_t is not None:
			dt = max(now - prev_t, 1e-3)
			dist_px = float(np.hypot(cx - prev_pt[0], cy - prev_pt[1]))
			scale = mpp if mpp is not None else DEFAULT_MPP
			dist_m = dist_px * scale
			speed_mps = dist_m / dt
			inst_speeds_mps.append(speed_mps)
		prev_positions_by_id[ID] = (cx, cy)
		last_seen_time_by_id[ID] = now
	if inst_speeds_mps:
		vehicle_count_instance.update_avg_speed(lane, float(np.median(inst_speeds_mps)))
	return prev_positions_by_id, last_seen_time_by_id

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()

ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
# video = pafy.new(inputVideoPath)
# best = video.getbest(preftype="mp4")
# url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# videoStream = cv2.VideoCapture(best.url)

# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees

class TrackCount:
	_instance = None

	def __init__(self):
		# Shared list across processes: [lane0, lane1, lane2, lane3]
		if not hasattr(self, "vehicle_lane_count"):
			self.vehicle_lane_count = multiprocessing.Manager().list([0,0,0,0])
			print(self.vehicle_lane_count)
		if not hasattr(self, "avg_speed_mps"):
			self.avg_speed_mps = multiprocessing.Manager().list([0.0,0.0,0.0,0.0])
		if not hasattr(self, "schedule"):
			self.schedule = multiprocessing.Manager().dict({
				'current_green_lane': 1,
				'time_left_sec': 0.0,
				'order': [1,2,3,4],
				'next_green_durations': [15,15,15,15]
			})

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def update_count(self,lane,value):
		self.vehicle_lane_count[lane] = value

	def reset_count(self,lane):
		self.vehicle_lane_count[lane] = 0

	def get_count(self,lane):
		return self.vehicle_lane_count[lane]

	def update_avg_speed(self,lane,value_mps):
		prev = float(self.avg_speed_mps[lane])
		alpha = 0.2
		self.avg_speed_mps[lane] = max(0.0, alpha*float(value_mps) + (1-alpha)*prev)

	def get_avg_speed(self,lane):
		return float(self.avg_speed_mps[lane])

	def set_schedule(self, current_green_lane, time_left_sec, order, next_green_durations):
		self.schedule['current_green_lane'] = int(current_green_lane)
		self.schedule['time_left_sec'] = float(time_left_sec)
		self.schedule['order'] = list(order)
		self.schedule['next_green_durations'] = list(next_green_durations)

	def get_schedule(self):
		return dict(self.schedule)
		
# loop over frames from the video file stream

def yolo_detection_counter(vehicle_count_instance,lane,inputVideoPath,outputVideoPath):
	videoStream = cv2.VideoCapture(inputVideoPath)
	fps = FPS().start()
	time.sleep(1.0)
	video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Specifying coordinates for a default line 
	x1_line = 0
	y1_line = video_height//2
	x2_line = video_width
	y2_line = video_height//2
	#Initialization
	previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
	prev_positions_by_id = {}
	last_seen_time_by_id = {}
	fvs = FileVideoStream(inputVideoPath).start()
	writer = initializeVideoWriter(video_width, video_height, videoStream,outputVideoPath)
	start_time = int(time.time())
	num_frames = 0
	try:
	# while True:
		while fvs.more():
			# print("================NEW FRAME================")
			# num_frames+= 1
			# print("FRAME:\t", num_frames)
			# Initialization for each iteration
			boxes, confidences, classIDs = [], [], [] 
			vehicle_crossed_line_flag = False 

			#Calculating fps each second
			start_time, num_frames = displayFPS(start_time, num_frames)
			# Read next frame from threaded stream; stop when depleted
			if not fvs.more():
				break
			frame = fvs.read()
			if frame is None:
				break
			# Ensure frame has 3 color channels (some decoders may return grayscale)
			if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			

			# construct a blob from the input frame and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes
			# and associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for i, detection in enumerate(output):
					# extract the class ID and confidence (i.e., probability)
					# of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > preDefinedConfidence:
						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
						box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
									
						#Printing the info of the detection
						#print('\nName:\t', LABELS[classID],
							#'\t|\tBOX:\t', x,y)

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# # Changing line color to green if a vehicle in the frame has crossed the line 
			# if vehicle_crossed_line_flag:
			# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0xFF, 0), 2)
			# # Changing line color to red if a vehicle in the frame has not crossed the line 
			# else:
			# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 0xFF), 2)

			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes; handle empty lists safely
			if len(boxes) and len(confidences):
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, float(preDefinedConfidence), float(preDefinedThreshold))
			else:
				idxs = []

			# Draw detection box 
			drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

			vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count_instance.get_count(lane), previous_frame_detections, frame)
			# Avoid runaway increments by capping per-frame new IDs
			vehicle_count = min(vehicle_count, 10_000)

			# Speed update & overlay (km/h only)
			mpp = None
			if meters_per_pixel is not None:
				mpp = float(meters_per_pixel[lane])
			prev_positions_by_id, last_seen_time_by_id = compute_and_update_speed(
				lane, current_detections, prev_positions_by_id, last_seen_time_by_id, mpp, vehicle_count_instance
			)

			avg_speed_mps = vehicle_count_instance.get_avg_speed(lane)
			avg_speed_text = None
			recommendation_text = None
			if display_advice:
				# Always show in km/h (use DEFAULT_MPP if calibration missing)
				avg_speed_text = f"Avg speed (km/h): {avg_speed_mps*3.6:.1f}"
			displayVehicleCount(frame, vehicle_count,lane, avg_speed_text, None)
			vehicle_count_instance.update_count(lane,vehicle_count)
			# cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			# 	(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
			# write the output frame to disk
			writer.write(frame)
			cv2.imshow('Frame',cv2.resize(frame,(1200,800)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break	
			fps.update()
			# Updating with the current frame detections
			previous_frame_detections.pop(0) #Removing the first frame from the list
			# previous_frame_detections.append(spatial.KDTree(current_detections))
			previous_frame_detections.append(current_detections)
			# if fvs.Q.qsize() <10:
			# 	fvs.stop()
			# 	fvs = FileVideoStream(inputVideoPath).start()
				#videoStream = cv2.VideoCapture(inputVideoPath)
				#writer = initializeVideoWriter(video_width, video_height, videoStream)
	except Exception as error:
		pass
	finally:
		fps.stop()
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		# do a bit of cleanup
		cv2.destroyAllWindows()
		fvs.stop()
		# release the file pointers
		print("[INFO] cleaning up...")
		writer.release()
		videoStream.release()
		return

if __name__ == '__main__':
	vehicle_count_instance = TrackCount()
	processes = []
	try:
		processes.append(multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,0,inputVideoPathList[0],outputVideoPathAll[0])))
		processes.append(multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,1,inputVideoPathList[1],outputVideoPathAll[1])))
		processes.append(multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,2,inputVideoPathList[2],outputVideoPathAll[2])))
		processes.append(multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,3,inputVideoPathList[3],outputVideoPathAll[3])))
		processes.append(multiprocessing.Process(target=traffic_control, args=(vehicle_count_instance, max_controller_seconds, max_controller_cycles)))

		for p in processes:
			p.start()
		for p in processes:
			p.join()
	except KeyboardInterrupt:
		print("\n[MAIN] KeyboardInterrupt received. Terminating processes...")
		for p in processes:
			if p.is_alive():
				p.terminate()
		for p in processes:
			p.join()
		print("[MAIN] Clean shutdown complete.")