#Module Declaration#
#------------------#
import os

import numpy as np
import pandas as pd
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import cv2
import os
from keras_facenet import FaceNet
from keras.models import load_model
import pickle
#------------------------------------#
faces=[]
facenet_model = FaceNet() #loading the facenet model
mtcnn_model = MTCNN()        #loading the mtcnn model
img = cv2.imread(r'./Good_images/chris hemsworth.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#extracting face from image
result = mtcnn_model.detect_faces(img)
#getting dimensions from mtcnn result and
#storing the cordinates
x1, y1, width, height = result[0]["box"]
x2, y2 = x1 + width, y1 + height
#cropping the image using the coordinates
img = img[y1:y2, x1:x2]
#appending the face extracted to a list
img = cv2.resize(img, (160, 160))
img = np.reshape(img, (1, 160, 160, 3))
face_feature = facenet_model.embeddings(img)
print(face_feature)
print(len(face_feature[0]))

model = load_model("./ANN_model/")
encoder = pickle.load(open("./encoder.pkl", "rb"))

prediction = model.predict(face_feature)[0] # preict probability vector for index no. 5
print('prediction probability for all possible values of data points rounded to 3 places:\n ',prediction.round(3))
if (max(prediction) > 0.95):
    accuracy = round(max(prediction) * 100, 2)
    prediction = np.argmax(prediction)
    print("max", prediction)
    label = encoder.inverse_transform([prediction])[0]
    answer = label
    label = "Name : {0} , Accuracy : {1}%".format(label, accuracy)
    print(label)