import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(1) 

lower = np.array([90, 70, 50])      #batas bawah kode biru
upper = np.array([130, 255, 255])   #batas atas kode biru

while True:
    _,img = webcam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    blue = cv.inRange(gray, lower, upper) #detect biru saja
    
    garis_kontur,_ = cv.findContours(blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    #nyari garis2 yang kedetect

    if len(garis_kontur) > 0: 
        for garis in garis_kontur:
            if cv.contourArea(garis) > 300: 
                #ngedetect bentuk yang di ataas 300 px

                x, y, w, h = cv.boundingRect(garis) 
                #dari bentuk ga jelas kita buatkan kotak tegak  lurus
                
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv.putText(img, 'Blue' ,(x-10, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    

    cv.imshow('Blue Detection', blue)
    cv.imshow('Normal', img)

    key = cv.waitKey(10)
    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()