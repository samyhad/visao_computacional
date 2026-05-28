import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        cv.imwrite('webcam_capture.jpg', frame)  # Save the captured frame as an image file
        print("Frame captured and saved as 'webcam_capture.jpg'. Exiting...")
        break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
