import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils #print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
#Application
import cv2
from keras.models import load_model
from collections import deque #queue

model1 = load_model('mnist.h5')        #Loaded saved model
#print(model)

letter_count = { 0: 'CHECK',                #Made dictionary to map output to corresponding letter
                 1: '0',
                 2: '1',
                 3: '2',
                 4: '3',
                 5: '4',
                 6: '5',
                 7: '6',
                 8: '7',
                 9: '8',
                 10: '9',
                 11: 'CHECK',
                 }

def keras_predict(model, image):        #Function to predict character on live web cam input
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    #print(pred_probab)
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):           #Process image from live web cam input
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))

    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

cap = cv2.VideoCapture(0)       #Instantiating object to read input from web cam
Lower_blue = np.array([110, 50, 50])
Upper_blue = np.array([130, 255, 255])      #Defined range for giving input
pred_class = 0
pts = deque(maxlen=512)
blackboard = np.zeros((480,640,3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
while(cap.isOpened()):
    ret , img = cap.read()
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, Lower_blue, Upper_blue)
    blur = cv2.medianBlur(mask, 15)
    blue = cv2.GaussianBlur(blur, (5,5), 0)         #Processed image for better read
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    center = None
    if len(cnts) >= 1:
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) > 250:
            ((x,y), radius) = cv2.minEnclosingCircle(contour)       #Made circles to use centers as coordinates for input
            cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(img, center, 5, (0,0,255), -1)
            M = cv2.moments(contour)
            center = (int(M['m10']/M['m00']), int(M['m01']/ M['m00']))      #Calculated center from moments
            pts.appendleft(center)
            for i in range(1, len(pts)):
                if(pts[i-1] is None or pts[i] is None):
                    continue
                cv2.line(blackboard, pts[i-1], pts[i], (255,255,255), 10)       #Draw line on blackboard and image from appended points
                cv2.line(img, pts[i-1], pts[i], (0,0,255), 5)
    elif len(cnts) == 0:
        if len(pts) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5,5), 0)       #Processed blackboard image for better read
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                #print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)      #Converted image to be processed in required format
                    digit = blackboard_gray[y-30:y+h+30, x-30:x+w+30]

                    # new image = process_letter(digit)

                    pred_probab, pred_class = keras_predict(model1, digit)
                    print(pred_class, pred_probab)
        pts = deque(maxlen=512)
        blackboard = np.zeros((480, 640, 3), dtype = np.uint8)
    cv2.putText(img, "Conv Network :" + str(letter_count[pred_class+1]), (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)      #Displayed character corresponding o input from dictionary
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k = cv2.waitKey(10)         #Use Escape key to exit program and close frames
    if k==27:
        break
