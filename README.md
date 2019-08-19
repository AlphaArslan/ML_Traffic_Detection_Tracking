# recognizing vehicles and traffic signs and tracking them
![demo.gif](demo.gif)

## Table of contents  
- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
    * [image_process.py](#image_process.py)
    * [seq_image_process.py](#seq_image_process.py)
    * [vid_process.py](vid_process.py)




## How it works
This code simply feeds [yolo](https://pjreddie.com/darknet/yolo/) trained model into __cv2__ deep neural network

```python
  net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
```
and track the detected objects using [sort](https://github.com/abewley/sort)

<p></p>

## Prerequisites
- download [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put it in _yolo-coco_ folder
- make sure you have these libs installed:
  * numpy
  * imutils
  * opencv

<p></p>

## Getting started
There is actually __three__ different script, each doing a distinguished task.
<p></p>

### image_process.py
This script is used to process images that are not necessarily taken in a sequence (images taken in different times). It only detects objects but no tracking is done.
<p></p>

Simply, put your images in _input/image/_.
Run
```bash
  python3 image_process.py
```
It will output:
* processed images (images with boxes and text) in _output_.
* in _det_results/_, text file for each image containing detected classes  in this format:
> classID  confidence  right  top  left  bottom  

<p></p>


### seq_image_process.py
This script is used to process images taken in a sequence (like video frames). It detects objects and track them.
<p></p>

Put your images in _input/image/_.
Run
```bash
  python3 seq_image_process.py
```
It will output:
* processed frames (images with boxes and text) in _output_.
* one __CSV__ file with a line format like follows:
> frame_index  DetectionIndex_ClassName  right  top  left  bottom

<p></p>

### vid_process.py
This script is used to process videos. It detects objects and track them.
<p></p>

Rename your video to __input.mp4__ and move it to _input/video_.
Run
```bash
  python3 vid_process.py
```
It will output:
* processed frames (images with boxes and text) in _output_.
* processed video in __AVI__ format named __output.avi__.
* one __CSV__ file with a line format like follows:
> frame_index  DetectionIndex_ClassName  right  top  left  bottom


<p></p>

## Citation

### YOLO :

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
