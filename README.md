# What is this?

This is a simple Machine Learning open computer vision
application that number plates in real time from an image
using `haarcascade_russian_plate_number.xml` from opencv.

### Demo
[](https://github.com/CrispenGari/car-plates-detection-open-computer-vision/blob/main/images/bandicam%202021-01-04%2012-15-41-143.jpg)
### App capabilities.
This app is cappable of:
* detecting car number plates  in real time
* draw rectangle around each plate detected
* put text label `Plate` to show that this is a Plate detected

### First of all install `opencv-python` and `numpy`:
#### You can install `opencv-python` by running:

`python -m pip install opencv-python`

#### You can install `numpy` by running:
`python -m pip install numpy`

### Alternatively, you can install all of them by pasting the following code on your `main.py` and run it.

````buildoutcfg
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
````

### After installation of all the packages then we are ready to go.

#### step 1:
Define a function that detects face, and accepts an image path as its argument.


#### step 2:
Load the face cascade
````buildoutcfg
 plateClassifier = cv2.CascadeClassifier("cascade/haarcascade_russian_plate_number.xml")
````
#### step 3:
Load an image and Convert the image to gray-scale and detects
plates-points using the `detectMultiScale()` function.
````buildoutcfg
image = cv2.imread(image_path)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plates = plateClassifier.detectMultiScale(grayImage, 1.5)
````
Now we have all the point to draw the rectangle to our original image.

#### step 4:
Detect plates and
Loop through plates array and draw a rectangle around each plate
that is going to be detected. 
Draw the text as was as a rectangle that is 
filled with green color that bounds the text
and show the image.
````buildoutcfg
if len(plates):
    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y - 20), (x + int(w / 2), y), (0, 255, 0), -1)
        cv2.putText(image,"Plate",(x + 10, y-5), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255))
        final_image = cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 1)
        cv2.imshow("Number Plates Detector",final_image)
````
Now everything is ready we are only left with showing images in a loop and wait 
for `q` key press to close the loop.
#### step 6:
````buildoutcfg
key = cv2.waitKey(0)
if key & 0xFF == ord('q'):
    break
````

### All code in one place: `main.py`

````buildoutcfg

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
````
### Where to find the face cascade classifier?
You will find it on the opencv github [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)
 
### Why this simple App?
This app was build for practise purposes.
