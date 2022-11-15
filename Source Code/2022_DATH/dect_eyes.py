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

##### Arguments #####
EYE_THRESH = 0.25
EYE_FRAMES = 25
COUNTER = 0
##### End : Arguments #####

##### Functions #####
def calc_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def results_EAR(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = calc_EAR(leftEye)
    rightEAR = calc_EAR(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)
 
def alarm(msg):
    print('call')
    s = 'espeak "'+ msg +'"'
    os.system(s)
           
def main():      
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened() :
        ret, frame = cap.read() 
        frame = imutils.resize(frame, width=500)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
    
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = results_EAR(shape)
            ear = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            
            
            if ear <= EYE_THRESH:
                COUNTER += 1
                print(COUNTER)
                if COUNTER >= EYE_FRAMES:   
                    print('Warning Sir') 
                    t1=threading.Thread(target=alarm, args=('wake up sir',))
                    # alarm('Wake up sir')     
                    t1.start()
                    t1.join()                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            else:
                COUNTER = 0
                alarm_status = False
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (380, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      
        
        cv2.imshow("Detect Driver State : EAR", frame)
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break        
    cap.release()
    cv2.destroyAllWindows()

##### End Functions #####


##### Execute Main Code #####
if __name__ == "__main__":
    main()