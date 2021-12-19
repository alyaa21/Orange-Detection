import cv2
import numpy as np

lowerBound = np.array([1, 100, 80])
upperBound = np.array([22, 256, 256])

font = cv2.FONT_HERSHEY_SIMPLEX
kernal=np.ones((2,2),np.uint8)
while True:
    img = cv2.imread('orange_pic.jpg')

    img = cv2.resize(img, (512,512))
    cv2.imshow("original", img)
    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # cv2.imshow('hh', mask)
    erosion=cv2.erode(mask, kernal, iterations=3)
    # cv2.imshow('mask', erosion)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for countours in conts:
        x, y, w, h = cv2.boundingRect(countours)
        if cv2.contourArea(countours) < 80:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("detect", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
