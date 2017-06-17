import cv2
import numpy as np

# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0, 133, 77],np.uint8)
max_YCrCb = np.array([255, 173, 127], np.uint8)

roi_rec_cords = ((100, 150), (500, 600))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #resizes to correct input

    x1, x2 = roi_rec_cords[0][0], roi_rec_cords[1][0]
    y1, y2 = roi_rec_cords[0][1], roi_rec_cords[1][1]
    roi_rec = frame[y1:y2, x1:x2] #numpy switches x and y..
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

    #Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(roi_rec, cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    im2, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour on the source image
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(roi_rec, contours, i, (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
