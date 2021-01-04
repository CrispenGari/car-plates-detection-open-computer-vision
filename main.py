
# Plates detector
""""
What is this?
    * This is a car number plates detector using open computer vision.
*   This app detects car plates from an image.
"""

try:
    import cv2
    import numpy as np
except ImportError as e:
    from pip._internal import main as install
    packages = ["numpy", "opencv-python"]
    for package in packages:
        install(["install", package])
finally:
    pass

def plateDetector(image_path):
    plateClassifier = cv2.CascadeClassifier("cascade/haarcascade_russian_plate_number.xml")
    image = cv2.imread(image_path)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load the cascade classifier
    plates = plateClassifier.detectMultiScale(grayImage, 1.5)
    if len(plates):
        for (x, y, w, h) in plates:
            cv2.rectangle(image, (x, y - 20), (x + int(w / 2), y), (0, 255, 0), -1)
            cv2.putText(image,"Plate",(x + 10, y-5), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255))
            final_image = cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 1)
            cv2.imshow("Number Plates Detector",final_image)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return
plateDetector("images/car2.jpg")
