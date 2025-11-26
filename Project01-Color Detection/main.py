import cv2

from util import get_limits

from PIL import Image

Yellow=[0,255,255]

webcam=cv2.VideoCapture(0)

ret=True

while ret:
    ret,frame=webcam.read()
    hsvImage=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit=get_limits(Yellow)
    mask=cv2.inRange(hsvImage,lowerLimit, upperLimit)
    mask1=Image.fromarray(mask)
    bbox=mask1.getbbox()
    if bbox is not None:
        x1,y1,x2,y2=bbox
        frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),5)
    cv2.imshow('Detector',frame)
    if cv2.waitKey(40) & 0xFF==ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()

