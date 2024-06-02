import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame',0)
cv2.resizeWindow('frame',300,300)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300,300))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        vidout=cv2.resize(frame,(300,300)) #create vidout funct. with res=300x300
        out.write(vidout) #write frames of vidout function
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
