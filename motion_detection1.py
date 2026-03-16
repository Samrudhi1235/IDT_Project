import cv2
import numpy as np

# load human detector
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# start webcam
cap = cv2.VideoCapture(0)

# read first frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    # motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # detect humans
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray_frame, 1.1, 3)

    # detect pen-like thin objects
    edges = cv2.Canny(gray_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)

    suspicious = False

    # draw human boxes
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)

    # check motion contours
    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue

        mx,my,mw,mh = cv2.boundingRect(contour)

        # check if motion is near human
        for (x,y,w,h) in bodies:
            if (mx > x-50 and mx < x+w+50 and my > y-50 and my < y+h+50):
                suspicious = True

        cv2.rectangle(frame1,(mx,my),(mx+mw,my+mh),(0,255,0),2)

    # check pen-like lines
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if length > 50:  # long thin object
                # cv2.line(frame1,(x1,y1),(x2,y2),(0,255,255),2)
                pass

                for (x,y,w,h) in bodies:
                    if (x1 > x-50 and x1 < x+w+50 and y1 > y-50 and y1 < y+h+50):
                        suspicious = True

    # alert
    if suspicious:
        cv2.putText(frame1,"Suspicious Action Detected",(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("Smart Street Light Monitoring", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()