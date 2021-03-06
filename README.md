## What is SAVSDN?
The Scene-Aware Video Spark Detection Network (SAVSDN) can detect high-speed flying sparks in visible light video, which consists of an information fusion-based cascading video codec-YOLOv5 structure. Because there are a lot of disturbances in the environment that have the characteristics of bright line segments that are exactly the same as sparks. Existing image object detectors and video object detectors have a large amount of over-detection for spark detection. The video codec in SAVSDN can be cascaded with any image object detector, and the codec can sense scenes and sparks through an improved sequence-to-sequence model. This video spark detection technology has been published in a journal article, and you can get more details from https://doi.org/10.3390/s21134453 .
### The three main difficulties in video spark intelligent detection:
![1625052610](https://user-images.githubusercontent.com/83768527/123953358-b69ff900-d9d9-11eb-8be8-a5766e9a96e3.png)

The figure shows the three difficulties faced by intelligent spark detection. First, the feature of a single spark is a bright line segment, which is completely consistent with the features of the lights, metal cables, and reflected light in the image. Second,a lack of abnormal spark data hinders the training of supervised learning models. Third, more than 10 cameras are used to monitor the status of aero engines, which requires the spark detection algorithm to be real-time and parallel.

It is worth noting that because the high-speed airflow in the aero engine test cabin will cause the camera to shake, the video images are often accompanied by jitter, which causes all the line segments in the scene to also move in complex time series.



### The network structure of SAVSDN：
![1625052649(1)](https://user-images.githubusercontent.com/83768527/123953362-b7d12600-d9d9-11eb-901f-8ae4b61c1ce3.png)

In fact, we proposed a real-time anomalous spark video detection method based on deep learning for aero engine testing. Each time, the input was a sequence of nine consecutive frames, and the output was the position of all the anomalous sparks in the last three frames. In particular, the first seven frames were used by SAVSDN to perceive the bright line segments interference in the scene, so that the network could easily distinguish the sparks and the interference. First of all, nine frames of image sequences were inputted for preprocessing, including image size normalization and image grayscale. Then, the input was divided into 4 groups of image sub-sequences in chronological order, and each group of image sub-sequence contained 3 consecutive images. Next, four weight-sharing ConvLSTM were used to extract motion feature images from the four image sub-sequences, respectively. Moreover, a new sequence will be created by integrating the feature images corresponding to the first three groups of image sub-sequences. In addition, the new sequence will be sent to another ConvLSTM to extract scene perception feature images, which contains long time existing interference of bright line segments. Moreover, after obtaining scene perception feature images, another ConvLSTM was used to integrate the scene perception feature images and the motion feature images corresponding to the last image sub-sequence, and was then decoded to obtain a spatio-temporal deep feature image. The spatio-temporal deep feature image contained the spatio-temporal features that could distinguish sparks from the interference of bright line segments, which will be sent to the subsequent image object detector to detect spark features again. If the image object detector detects a spark, the human-computer interaction terminal will alarm and continuously monitor, otherwise continue to monitor.



## What is SAES?
As far as we know, the Simulated Aero Engine Spark (SAES) data set is the first published data set for aero engine anomalous spark detection. The data set contains 26,382 images and correlated spark tags in total. The images come from a video of 17 minutes and 35 seconds in length, captured by a Hikvision DS-2CD3T45FP1-LS camera with a frame rate of 25 fps. All the images were taken from a simulation aero engine chamber, which we constructed. The inner chamber comprised a set of complex interferences, including complex illumination changes, colorful shaking cables, aero engine vibration, flickering flames, unsteady video interface, and shining metal surfaces.
### The simulated aero engine chamber:

![image](https://user-images.githubusercontent.com/55908288/120090714-256a0800-c137-11eb-93e3-2e5e2386db3a.png)

![image](https://user-images.githubusercontent.com/55908288/120090721-29962580-c137-11eb-8d09-831120f07a21.png)


### The SAES dataset samples:

![image](https://user-images.githubusercontent.com/55908288/120090586-5a298f80-c136-11eb-9056-83f2e2c597a3.png)

![image](https://user-images.githubusercontent.com/55908288/120090589-5f86da00-c136-11eb-8b01-eedaa969d269.png)

![image](https://user-images.githubusercontent.com/55908288/120090593-631a6100-c136-11eb-9a6b-900a10713d99.png)


## The link to SAES data set and SAVSDN's detection results:
```
https://pan.baidu.com/s/1A7ocxF9LS8GrKGlJFQYpgA   (Extraction code:o0rc)
```
## How to run the source code for spark detection?
The source code of the video codec we proposed is "models/autoCodeNet.py" and a trained model file that can directly detect sparks is "weights/sparkDetectYolov5s/epoch17.pt".
1. Install the necessary environment:
```
pip install -r requirements.txt
```
2. Save the video to be detected to the "video/" directory.
3. Start spark detection:
```
python detect.py
```
4. The test results will be saved to the "runs/detect/exp/" directory.
## Test
SAVSDN is tested in a simulated environment and some real scenes.

![image](https://user-images.githubusercontent.com/55908288/120088969-a40b7900-c128-11eb-9265-f8be2c43f511.png) ![image](https://user-images.githubusercontent.com/55908288/120088975-b4235880-c128-11eb-8523-68c936d03faa.png)

![image](https://user-images.githubusercontent.com/55908288/120088977-c00f1a80-c128-11eb-8e48-47279c8a642f.png) ![image](https://user-images.githubusercontent.com/55908288/120088979-c43b3800-c128-11eb-8402-22f2aba17e2a.png)


When we apply it to the simulated aero engine chamber, the result is good enough and almost exists no mistake.

![sparks](https://user-images.githubusercontent.com/55908288/120089937-5a735c00-c131-11eb-81c3-bf0d9bda7df9.gif)

## Development
SAVSDN can also be applied to other fields with perfect performance.

### Other sparks:

![other_sparks_1 00_00_00-00_00_08](https://user-images.githubusercontent.com/55908288/120110521-6a765480-c1a0-11eb-8cf4-2324d974d29b.gif)

![other_sparks_2 00_00_00-00_00_10](https://user-images.githubusercontent.com/55908288/120110530-706c3580-c1a0-11eb-9a42-23ebedb5bd71.gif)


### Badminton:

![04_badminton 00_00_00-00_00_30](https://user-images.githubusercontent.com/83768527/120094158-6111cb80-c151-11eb-95cd-98503b5c8c3b.gif)

![badminton2 00_00_02-00_00_19](https://user-images.githubusercontent.com/55908288/120110543-77934380-c1a0-11eb-809c-0515827e8185.gif)

![badminton3 00_00_02-00_00_12](https://user-images.githubusercontent.com/55908288/120110553-7cf08e00-c1a0-11eb-89b9-3ba0b78cc153.gif)


### Rubbish thrown from upstairs:

![07_high-altitude parabolics 00_00_00-00_00_30](https://user-images.githubusercontent.com/83768527/120094295-0fb60c00-c152-11eb-9964-b961017c982e.gif)


### Shooting:

![03_tracer bullets 00_00_00-00_00_30](https://user-images.githubusercontent.com/83768527/120094306-1e042800-c152-11eb-9ac0-e8a53cd3aef9.gif)


## References
YOLOv5 (https://github.com/ultralytics/yolov5) is the image object detector in SAVSDN of this repository.

## Acknowledgements
This research is commissioned by AECC (Aero Engine Corporation of China) Sichuan Gas Turbine Establishment.

## Details
This video spark detection technology has been published in a journal article, and you can get more details from https://doi.org/10.3390/s21134453 .

If you have any question, please contact us by emailing to kj317099295@stu.xjtu.edu.cn.


