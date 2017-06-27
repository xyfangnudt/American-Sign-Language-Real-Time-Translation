import cv2
import numpy as np

roi_rec_cords = ((100, 100), (500, 600))

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(0, 7)

bg_captured = False
skin = None


def capture_background(frame):
    fgmask = fgbg.apply(frame)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


while True:
    ret, frame = cap.read()
    #resizes to correct input
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    x1, x2 = roi_rec_cords[0][0], roi_rec_cords[1][0]
    y1, y2 = roi_rec_cords[0][1], roi_rec_cords[1][1]
    roi_rec = frame[y1:y2, x1:x2]  # numpy switches x and y..

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

    if bg_captured:

        skin = capture_background(roi_rec)
        # blur_skin = cv2.GaussianBlur(skin, (41, 41), 0)
        _, thresh = cv2.threshold(skin, 10, 255, cv2.THRESH_BINARY)

        cv2.imshow('skin', skin)



    # cv2.imshow('skin', skin)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break
    elif k == -1:
        pass
    elif k == 99:
        bg_captured = True
    else:
        print(k)



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()