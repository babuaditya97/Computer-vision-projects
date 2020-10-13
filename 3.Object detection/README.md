# Object-Detection using pretrained yoloV3 model

This project implements an image and video object detection classifier using pretrained yolov3 model. 
The yolov3 model is taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet).


![yolo_infer_3](https://user-images.githubusercontent.com/26242097/48850729-449db700-edcf-11e8-853d-9f3eca6233c9.png)
![yolov3-video](https://user-images.githubusercontent.com/26242097/48851021-0785f480-edd0-11e8-8ce4-cdfb78e8a849.png)



## References

1) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
2) https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV


### Requirements
Run the script to install all nessasary packages

```sh
$ pip intsall requirements.txt
```


### Code flow
add path of the video file to be processed in line 16 
run " yolo_detect_video.py "

Detailed description along with the model bulding and analysis can be found in the code file.


