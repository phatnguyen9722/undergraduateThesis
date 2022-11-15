import uuid
import cv2
import os
import time

IMAGES_PATH = os.path.join('data','images')
labels = ['awake','drowsy']
NUMBER_IMAGES = 10 

cap = cv2.VideoCapture(0)
for label in labels : 
    print('Collecting Images {}'.format(label))
    time.sleep(5)

    for img_num in range(NUMBER_IMAGES) : 
        print('collecting images for {}, images number : {}'.format(label,img_num))

        ret, frame = cap.read()

        imgname = os.path.join(IMAGES_PATH,label+'.'+str(uuid.uuid1())+'.jpg')
        
        #write out images to file 
        cv2.imwrite(imgname,frame)
        cv2.imshow('Image Collector',frame)
        
        # 2 second to delay
        time.sleep(2)

    if cv2.waitKey(10) & 0xFF == ord('q') : 
        break

cap.release()
cv2.destroyAllWindows()


