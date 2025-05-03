from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("./Desarrollo/simulation/Env04/models_params_weights/YOLO/yolo11m-seg.pt") #"yolov8n-seg.pt"

img = cv2.imread("./Desarrollo/simulation/Env04/DataSets/RawTools/calibre01.png")
#img = cv2.imread("./Desarrollo/simulation/Env04/DataSets/RawTools/pinza_chica01.png")
print(img.shape)

# if you want all classes
#yolo_classes = list(model.names.values())
#classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

results = model.predict(img, conf=0.5, device='cuda:0', imgsz=640) # to limit the max quantity of detections: max_det
#colors = [random.choices(range(256), k=3) for _ in classes_ids]
#print(mask)
bw_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # single channel, 8-bit
for result in results:
    try:
        for mask in result.masks.xy:
            points = np.int32([mask])
            cv2.fillPoly(bw_mask, points, 255)
    except:
        pass
        

cv2.imshow("Image", bw_mask)
cv2.waitKey(5000)
print(bw_mask.shape)