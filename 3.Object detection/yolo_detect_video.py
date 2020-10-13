import cv2
import numpy as np


''' loading and setting up the pretrained yoloV3 '''

#loading model weights & configuration file
net = cv2.dnn.readNet('utils/yolov3.weights','utils/yolov3.cfg')
#loading target classes
classes = []
with open('utils/coco.names','r') as f:
    classes = f.read().splitlines()
print(classes)

# loading input video
cap = cv2.VideoCapture('samp2.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")


# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))############ for saving the video


'''reading frame by frame and detecting'''
while True:
    try:
        _, img = cap.read()

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 :
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colours = np.random.uniform(0,255, size = (len(boxes), 3))

        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w ,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                colour = colours[i]
                cv2.rectangle(img,(x,y), (x+w, y+h), colour,2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
                # out.write(img)############################# for saving the video
    except:
        pass
    
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
