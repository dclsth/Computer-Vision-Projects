import cv2 as cv
import numpy as np

def distance_ball(x1,x2,y1,y2):
    return (x1 - x2)**2+(y1-y2)**2

webcam = cv.VideoCapture(1)
prev_ball = None


while True:
    _, img = webcam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (17,17), 0)
                                             #jarak antar ball(2)  #kesempurnaannya(4)
    ball = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 2.7, 100, param1=280, param2=90, minRadius=5, maxRadius=400)
                                          #titik tengah(1)  #garis ball(3)

    if ball is not None:
        ball = np.uint16(np.around(ball))

        chosen_ball = None

        for i in ball[0, :]:
            if chosen_ball is None:
                chosen_ball = i
            if prev_ball is not None:
                if distance_ball(chosen_ball[0], chosen_ball[1], prev_ball[0], prev_ball[1]) <= distance_ball(i[0], i[1], prev_ball[0], prev_ball[1]):
                    chosen_ball = i  
        
        cv.circle(img, (chosen_ball[0], chosen_ball[1]), 1, (0,0,255), 2)
        cv.circle(img, (chosen_ball[0], chosen_ball[1]), chosen_ball[2], (0,255,0), 2)
        prev_ball = chosen_ball

    cv.imshow('BALL DETECTION', img)

    key = cv.waitKey(10)

    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()