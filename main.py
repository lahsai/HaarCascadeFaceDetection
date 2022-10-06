import cv2

cascadePath = "./haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascadePath)

videoCapture = cv2.VideoCapture(0)

if not videoCapture.isOpened:
    print("Error opening video capture")
    exit(0)

while True:
    # Capture frame and store it in the frame variable
    ret, frame = videoCapture.read()

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all the faces and store it in the faces variable
    # First argument is the grayscale image, second is the scale of the image, 1.25 reduces the image size by 25%
    # The third argument is the number of neighboring detections, increasing this value helps reduce multiple detections
    # of the same image object
    faces = cascade.detectMultiScale(gray, 1.25, 10)

    # Loop through each face in the faces variable
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face to the display output (the window on your screen)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw the captured frame to the display output
    # "Face Detection" is the window's title
    cv2.imshow("Face Detection", frame)

    # Quit when escape is pressed (key code 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture
videoCapture.release()

# Destroy all windows
cv2.destroyAllWindows()
