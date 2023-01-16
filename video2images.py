import numpy as np
import cv2 as cv

cont = 0
cap = cv.VideoCapture('videos\dataset_limpio_jon.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', frame)
    cv.imwrite(f'Output-Jon/1-{cont}.jpg', frame)
    if cv.waitKey(1) == ord('q'):
        break
    cont += 1
cap.release()
cv.destroyAllWindows()