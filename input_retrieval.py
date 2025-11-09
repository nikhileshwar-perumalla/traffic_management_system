import argparse
import os

# PURPOSE: Parsing the command line input and extracting the user entered values
# PARAMETERS: N/A
# RETURN:
# - Labels of COCO dataset
# - Path to the weight file
# - Path to configuration file
# - Path to the input video
# - Path to the output video
# - Confidence value
# - Threshold value
def parseCommandLineArguments():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input",
		help="path to input video")
	ap.add_argument("-o", "--output",
		help="path to output video")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-iall", "--inputall",nargs='+',type=str, help="input all 4 files ")
	ap.add_argument("-outputall", "--outputall",nargs='+',type=str, help="outpu all 4 files ")
	ap.add_argument("-t", "--threshold", type=float, default=0.2,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-u", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")

	# Optional: real-world calibration and advisory settings
	ap.add_argument("--meters-per-pixel", "--mpp", nargs='+', type=float, default=None,
		help="Meters per pixel for each of the 4 lanes (provide 4 values). If omitted, speeds are shown in px/s.")
	ap.add_argument("--distance-km", "--dkm", type=float, default=1.0,
		help="Distance upstream (in km) from which to compute recommended approach speed. Default 1.0 km")
	ap.add_argument("--speed-limit-kmh", "--sl", nargs='+', type=float, default=[60,60,60,60],
		help="Speed limit (km/h) for each lane, used to cap recommendations. Provide 4 values. Default 60 for all.")
	ap.add_argument("--display-advice", action='store_true', default=True,
		help="Overlay average and recommended speeds on the output frames.")

	# Graceful stop options
	ap.add_argument("--max-controller-seconds", type=int, default=0,
		help="Maximum seconds to run the traffic controller before auto-shutdown (0 = unlimited)")
	ap.add_argument("--max-controller-cycles", type=int, default=0,
		help="Maximum number of green phases to run before auto-shutdown (0 = unlimited)")

	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	
	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov7-tiny.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov7-tiny.cfg"])
	
	inputVideoPath = args["input"]
	inputVideoPathList = args["inputall"]
	outputVideoPath = args["output"]
	outputVideoPathAll = args["outputall"]
	confidence = args["confidence"]
	threshold = args["threshold"]
	USE_GPU = args["use_gpu"]

	meters_per_pixel = args.get("meters_per_pixel")
	distance_km = args.get("distance_km", 1.0)
	speed_limit_kmh = args.get("speed_limit_kmh", [60,60,60,60])
	display_advice = args.get("display_advice", True)

	# Normalize lengths
	if inputVideoPathList and len(inputVideoPathList) != 4:
		raise ValueError("--inputall must contain exactly 4 video paths (one per lane)")
	if outputVideoPathAll and len(outputVideoPathAll) != 4:
		raise ValueError("--outputall must contain exactly 4 output paths (one per lane)")
	if meters_per_pixel is not None and len(meters_per_pixel) != 4:
		raise ValueError("--meters-per-pixel must contain exactly 4 numeric values (one per lane)")
	if speed_limit_kmh and len(speed_limit_kmh) != 4:
		raise ValueError("--speed-limit-kmh must contain exactly 4 numeric values (one per lane)")

	max_controller_seconds = args.get("max_controller_seconds", 0)
	max_controller_cycles = args.get("max_controller_cycles", 0)

	return (LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence,\
			threshold, USE_GPU, inputVideoPathList, outputVideoPathAll, meters_per_pixel,\
			distance_km, speed_limit_kmh, display_advice, max_controller_seconds, max_controller_cycles)
