import cv2 as cv

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv.CascadeClassifier('haarcascade_eye.xml')
mouth = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
webcam = cv.VideoCapture(1)

while True:
    ret, frame = webcam.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.33,4)
    for(x, y, w, h) in faces:
        if len(faces) <= 1 :
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye.detectMultiScale(roi_gray, 1.3, 4)

            for(ex, ey, ew, eh) in eyes:
                if len(eyes) <= 2:
                    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,0,255), 2)
                else:
                    continue
            
            y_mouth = h//2 #tinggi mulut kedeteksi
            roi_gray1 = roi_gray[y_mouth:h, 0:w] #area buat detect mulut
            mouths = mouth.detectMultiScale(roi_gray1, 1.3, 4)

            for(mx, my, mw, mh) in mouths:
                if len(mouths) <= 1:
                    rect_y = my + y_mouth #tinggi mulut dari face
                    cv.rectangle(roi_color, (mx, rect_y), (mx + mw, rect_y + mh), (255,0,0), 2)
                else:
                    continue
        else:
            continue   
            
    cv.imshow("Face Detection", frame)

    key = cv.waitKey(10)
    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()