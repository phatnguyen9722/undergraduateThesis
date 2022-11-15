import torch
import cv2
import numpy as np

# # Model
model = torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp5/weights/last.pt',force_reload=True)

print("Starting camera")
cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    ret, frame = cap.read() 
    results = model(frame)
    cv2.imshow('Detect', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()

