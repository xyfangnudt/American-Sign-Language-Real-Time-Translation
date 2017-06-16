import cv2
from train import width, height

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #resizes to correct input

    resized = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
