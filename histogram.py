import cv2
import numpy as np
from matplotlib import pyplot as plt

def require_frame(video_path):
    cap = cv2.VideoCapture(video_path)  
    success,image=cap.read()
    count=0
    success=True
    while success:
        success,image = cap.read()
        cv2.imwrite("frame%d.jpg" % count, image)

        count += 1


def histogram_equalization(image):
    img = cv2.imread(image)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", img)
    result = cv2.equalizeHist(img)
    cv2.imshow("equalized", result)
    cv2.waitKey(0)
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    hist1 = cv2.calcHist([result], [0], None, [256], [0, 255])
    plt.figure()
    plt.plot(hist,color='red')
    plt.plot(hist1,color='green')
    plt.xlim([0, 256])
    plt.title('equalized Histogram')
    plt.show()
    


require_frame('/Users/lavinia/Desktop/eplipsy/08/00.mp4')
histogram_equalization("frame98.jpg")