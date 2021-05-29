# SAVSDN-SAES-SAVSDN_detection_result
## What is SAVSDN?
A Scene-Aware Video Spark Detection Network (SAVSDN) can detect high-speed flying sparks in visible light video, which consists of an information fusion-based cascading video codec-YOLOv5 structure. Existing image object detectors and video object detectors have a large amount of over-detection for spark detection. The video codec in SAVSDN can be cascaded with any image object detector, and the codec can sense scenes and sparks through an improved sequence-to-sequence model.
## What is SAES?
As far as we know, Simulated Aero Engine Spark (SAES) data set is the first published data set for aero engine anomalous spark detection. The data set contains 26,382 images and correlated spark tags in total. The images come from a video of 17 minutes and 35 seconds in length, captured by a Hikvision DS-2CD3T45FP1-LS camera with a frame rate of 25 fps. All the images were taken from a simulation aero engine chamber, which we constructed. The inner chamber comprised a set of complex interferences, including complex illumination changes, colorful shaking cables, aero engine vibration, flickering flames, unsteady video interface, and shining metal surfaces.
## The links to SAES data set and SAVSDN's detection results:
```
https://pan.baidu.com/s/1A7ocxF9LS8GrKGlJFQYpgA   (Extraction code:o0rc)
```
## How to run the source code for spark detection?
The source code of the video codec we proposed is "models/autoCodeNet.py". A trained model file that can directly detect sparks is "weights/sparkDetectYolov5s/epoch17.pt".
1. Install the necessary environment:
```
pip install -r requirements.txt
```
2. Store the video to be detected in the "video/" directory.
3. Start spark detection:
```
python detect.py
```
4. The test results will be saved to the "runs/detect/exp/" directory.
## Reference
YOLOv5 (https://github.com/ultralytics/yolov5) is the image object detector in SAVSDN of this repository.