import cv2 as cv
import winsound as ws
import threading as tr

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv.CascadeClassifier('haarcascade_eye.xml')
mouth = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
webcam = cv.VideoCapture(1)


def sound():
    ws.Beep(2500,300)

while True:
    ret, frame = webcam.read()
    face_num = 1
    eyes_num = 1
    mouth_num = 1
    jumlah_mata = 0
    jumlah_muka = 0
    jumlah_mulut = 0
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.33,4)
         
    for(x, y, w, h) in faces:
        t = tr.Thread(target=sound)
        t.start()
        jumlah_muka = len(faces)
        

        cv.putText(frame, f"Face {face_num}", (x-20,y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


        cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_gray, 1.3, 4)
        face_num+=1

        if len(eyes) <= 2:
            jumlah_mata += len(eyes)
            for(ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,0,255), 2)
                cv.putText(frame, f"Eyes {eyes_num}", (x + ex-10, y + ey-10), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1)
            
                eyes_num += 1

        y_mouth = h//2 #tinggi mulut kedeteksi
        roi_gray1 = roi_gray[y_mouth:h, 0:w] #area buat detect mulut
        mouths = mouth.detectMultiScale(roi_gray1, 1.3, 4)

        for(mx, my, mw, mh) in mouths:
            jumlah_mulut += len(mouths)
            if len(mouths) <= 1:
                rect_y = my + y_mouth #tinggi mulut dari face
        
                cv.putText(frame, f"Mouth {mouth_num}", (x+ mx-20, rect_y +y+ my-35), cv.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,0), 1)
                cv.rectangle(roi_color, (mx, rect_y), (mx + mw, rect_y + mh), (255,0,0), 2)
                mouth_num += 1

            else:
                continue
    cv.putText(frame, f"{jumlah_muka} Face Detected", (30,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv.putText(frame, f"{jumlah_mata} Eyes Detected", (30,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv.putText(frame, f"{jumlah_mulut} Mouth Detected", (30,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    cv.imshow("Face Detection", frame)

    key = cv.waitKey(10)
    if key == 27:
        break

webcam.release()
cv.destroyAllWindows()
