import dlib
import cv2
import imutils
import numpy as np
import time
import os
from threading import Thread
from scipy.spatial import distance as dist
from imutils import face_utils

from threading import Thread
import threading

YAWN_THRESH = 25

def alarm(msg):
    print('call')
    s = 'espeak "'+ msg +'"'
    os.system(s)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def main():
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)
    while cap.isOpened(): 
        ret, frame = cap.read() 
        frame = imutils.resize(frame, width=500)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
         #for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            distance = lip_distance(shape)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)  

            if (distance > YAWN_THRESH):
                    cv2.putText(frame, "Yawn Alert", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if alarm_status2 == False :
                        alarm_status2 = True
                        t2 = Thread(target=alarm, args=('take some fresh air sir',))
                        t2.start()
                        t2.join
            else:
                alarm_status2 = False
                
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Detect Yarn", frame)
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break        
    cap.release()
    cv2.destroyAllWindows()
        
##### Execute Main Code #####
if __name__ == "__main__":
    main()