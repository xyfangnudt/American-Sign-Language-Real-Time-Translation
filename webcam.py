import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #resizes to correct input
    cv2.imshow('frame', frame)
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
