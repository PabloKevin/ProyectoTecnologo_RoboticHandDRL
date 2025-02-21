from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("./Desarrollo/simulation/Env03/yolo11n-seg.pt") #"yolov8n-seg.pt"

#img = cv2.imread("./Desarrollo/simulation/Env03/DataSets/RawTools/martillo02.png")
img = cv2.imread("./Desarrollo/simulation/Env03/DataSets/RawTools/pinza_chica01.png")


# if you want all classes
#yolo_classes = list(model.names.values())
#classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.05

results = model.predict(img, conf=conf, device='cuda:0', imgsz=640)
#colors = [random.choices(range(256), k=3) for _ in classes_ids]
#print(mask)
bw_mask = np.zeros((255, 255), dtype=np.uint8)  # single channel, 8-bit
for result in results:
    for mask in result.masks.xy:
        points = np.int32([mask])
        cv2.fillPoly(bw_mask, points, 1)
        

cv2.imshow("Image", bw_mask)
cv2.waitKey(5000)