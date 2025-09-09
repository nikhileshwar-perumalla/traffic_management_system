# Adaptive Traffic Controller with YOLO Object Detection

## Overview
You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. For more detailed working of YOLO algorithm, please refer to the [YOLO paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).
Archive Paper : https://arxiv.org/abs/2207.02696 

This project aims to count every vehicle (motorcycle, bus, car, cycle, truck, train) detected in the input video using YOLOv7 object-detection algorithm and uses a real time Traffic control system to optimze the traffic signaling so as to lessen the impact on congestion and control emission. It takes 6 lane traffic and shows how we can route traffic in an optimized and envinormnet friendly way.
Object detection occurs at 4 lanes in real time the Main frame Central hub runs the detection and updates the vehicles count and runs ATCS.

## Authors
- [Karthik Rao](https://github.com/karthikrao117)
- [Sachin Kumar]()
- [Ashima Dujeja]()
- [Gaurav Keshwani]()



## Adaptive Traffic Control Sytem Network
 <p align="center">
  <img src="https://github.com/karthikrao117/Adaptive-Traffic-Controller-with-YOLO-Object-Detection/blob/main/example_gif/ATCS_system.jpg">
</p>

All the videos are run simultaneously as shown above and the ATCS is run in real time 

This project is enhancement of https://github.com/guptavasu1213/Yolo-Vehicle-Counter which was only used for counting vehicles.  

**We have used YOLO V7 which is 120% better than yolov5 itself and far more performant the YOlov3**.

**The read frames of opencv is slow so we have used imutils FileVideoStream which internally used Threads to read the vidoe frame in optimized way and to save the frames to queue which is 52% faster than cv2.read()**

**Since we are calculating the traffic in the lane when the signal goes red, the traffics slows down so we can have used the skip frame technique  to get better fps and no need to run the object detection on each frames the last few frames before the count is called is sufficient**

**This project can be run easily on CPU as the algorithms are optimised no need for GPU**
**This project can be run with live stream video data , as to say the direct video stream from traffic mounted , url path can be mentioned in the inputall args**



## Yolo V7 Performance
 <p align="center">
  <img src="https://github.com/karthikrao117/Adaptive-Traffic-Controller-with-YOLO-Object-Detection/blob/main/example_gif/yolov7.png">
</p>

## Working 
<p align="center">
  <img src="https://github.com/karthikrao117/Adaptive-Traffic-Controller-with-YOLO-Object-Detection/blob/main/example_gif/ATCS.png">
</p>
As shown in the image above, when the vehicles in the frame are detected, they are counted. After getting detected once, the vehicles get tracked and do not get re-counted by the algorithm. 

You may also notice that the vehicles will initially be detected and the counter increments, but for a few frames, the vehicle is not detected, and then it gets detected again. As the vehicles are tracked, the vehicles are not re-counted if they are counted once. 
Once the counting is done on each side when the traffic is red signal on each the next optimised lane for green signal is calcualted this takes in the vehicle count and the stravation lane into consideration while calculating the next lane.


## Prerequisites
* A 4 street video file to run the vehicle counting 
* The pre-trained yolov7 weight file should be downloaded by following these steps:
```
cd yolo-coco
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights
``` 

## Dependencies for using CPU for computations
* Python3 (Tested on Python 3.6.9)
check the official website for the right one matching your system 32 bit /64bit
```
https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe
```
* Pip
```
python -m pip install --upgrade pip
```
* OpenCV 3.4 or above(Tested on OpenCV 3.4.2.17)
```
pip install opencv-python==3.4.2.17
```
* Imutils 
```
pip install imutils
```
* Scipy
```
pip install scipy
```

## Dependencies for using GPU for computations
* Installing GPU appropriate drivers by following Step #2 in the following post:
https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/
* Installing OpenCV for GPU computations:
Pip installable OpenCV does not support GPU computations for `dnn` module. Therefore, this post walks through installing OpenCV which can leverage the power of a GPU-
https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/
## Usage
* `--inputall` or `-iall` argument requires the path to the input videos 4 video path separated by space 
* `--input` or `-i` argument requires the path to the input video
* `--output` or `-o` argument requires the path to the output video
* `--outputall` or `-oall` argument requires the path to the output video 4 output files separated by space
* `--yolo` or `-y` argument requires the path to the folder where the configuration file, weights and the coco.names file is stored
* `--confidence` or `-c` is an optional argument which requires a float number between 0 to 1 denoting the minimum confidence of detections. By default, the confidence is 0.5 (50%).
* `--threshold` or `-t` is an optional argument which requires a float number between 0 to 1 denoting the threshold when applying non-maxima suppression. By default, the threshold is 0.3 (30%).
* `--use-gpu` or `-u` is an optional argument which requires 0 or 1 denoting the use of GPU. By default, the CPU is used for computations
```
python yolo_video.py --inputall <input video paths> --output <output video path> --yolo yolo-coco [--confidence <float number between 0 and 1>] [--threshold <float number between 0 and 1>] [--use-gpu 1]
```
Examples: 
* Running with defaults
```
python yolo_video.py --inputall file1.mp4 file2.mp4 file3.mp4 file4.mp4  --outputall outfile1.avi outfile2.avi outfile3.avi outfile3.avi --yolo yolo-coco
```
* Specifying confidence
```
python yolo_video.py --inputall file1.mp4 file2.mp4 file3.mp4 file4.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --confidence 0.3
```
* Using GPU
```
python yolo_video.py --inputall file1.mp4 file2.mp4 file3.mp4 file4.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --use-gpu 1
```

## Implementation details
* The detections are performed on each frame by using YOLOv7 object detection algorithm and displayed on the screen with bounding boxes.
* The detections are filtered to keep all vehicles like motorcycle, bus, car, cycle, truck, train. The reason why trains are also counted is because sometimes, the longer vehicles like a bus, is detected as a train; therefore, the trains are also taken into account.
* The center of each box is taken as a reference point (denoted by a green dot when performing the detections) when track the vehicles.   
* Also, in order to track the vehicles, the shortest distance to the center point is calculated for each vehicle in the last 10 frames. 
* If `shortest distance < max(width, height) / 2`, then the vehicles is not counted in the current frame. Else, the vehicle is counted again. Usually, the direction in which the vehicle moves is bigger than the other one. 
* For example, if a vehicle moves from North to South or South to North, the height of the vehicle is most likely going to be greater than or equal to the width. Therefore, in this case, `height/2` is compared to the shortest distance in the last 10 frames. 
* As YOLO misses a few detections for a few consecutive frames, this issue can be resolved by saving the detections for the last 10 frames and comparing them to the current frame detections when required. The size of the vehicle does not vary too much in 10 frames and has been tested in multiple scenarios; therefore, 10 frames was chosen as an optimal value.
* Once the count is calculated at each lane the the ATCS algorithm is called which provides the next optimised green signal lane , The existing traffic system runs in round robin fashion with a fixed amount of time says 60s for each lane regardless of the number of vehicles present. Our algorithm fetches the real time vehicle count and and optimizes the traffic signal routing.
* The max(VEHICLE COUNT of all lanes) is calculated and is run for set period of time min(number of avialable vehicle in that most congested lane,60 sec) 
* That lane is made green rest are red and all the 3 ways are allowed left ,right ,forward 
* Once the orange light is reached the object detection is run again and the values are calualted and above 2 steps are run taking into consideration of starvation of lanes. All the lanes will get green signal once and completes a round robin but with prioritizing the congesion. if one lane is run once will not be run again unless all lanes are green signaled atleast once in a loop. This is done so as to prevent starvation of lane. Imagine if one lane has only 4-5 vehicles and the rest at all time have more than 10 the optimized counting will always return the congested one this will not allow the less congested to wait longer or never pass.
This can be averted with the average wait time at each lane is calcualted and once certain threshold of the time is reached the lane can be allowed.

## Future Enhancement 
* The traffic camera is expected to angled properly so as to represent only one lane as seen in the working screenshot the traffic that is leaving and not part of the lane is captured. This can be fixed while calculating the bounding boxes the position of the object is provided and the not since we are storing the 10th consecutive frame the previous position of that ID box and current position is provided know using k nearest neighbour algorithm so we based on the movement towards the origin (x,y)->(0,0) we can say that we need to count this vehicle and the others moving away we should'nt count it.
* The Average waiting time of each individual object(vehicle) can be calcualted and added as a paramter while calcuating the optimized traffic signal route since currently only the vehicle count is used and the round robin prioirty is done with average waiting time we can optimize this further

## Reference
* https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ 
* https://github.com/AlexeyAB/darknet
* https://github.com/WongKinYiu/PyTorch_YOLOv4
* https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose
* https://viso.ai/deep-learning/yolov7-guide/
* https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
* https://www.efftronics.com/adaptive-traffic-control-system#:~:text=ATCS%20algorithm%20adjusts%20traffic%20signal,congestion%20by%20creating%20smoother%20flow.
