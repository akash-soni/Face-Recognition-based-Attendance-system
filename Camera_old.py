import cv2
import sys
import pickle
import cvlib as cv
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from keras.models import load_model
from Frame_prediction import Frame_prediction
import sys

class Camera:
    def __init__(self,path, source):
        try:
            self.path = path
            self.face_feature_extractor = FaceNet()
            self.model = load_model(self.path + "/ANN_model/")
            self.encoder = pickle.load(open(self.path + "/encoder.pkl","rb"))
            self.source = source
        except Exception as e:
            print(e)
    

    def package(self):
        try:
            
            camera = []
            final_labels = []
            buffer_labels = {}
            no_of_frame = 0
            for i in self.source:
                buffer_labels[str(i)] = []
                cam = cv2.VideoCapture(i)
                camera.append(cam)
            while True:
                for i in range(len(camera)):
                    success , frame = camera[i].read()
                    if(success):
                        frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
                        frame, label  = Frame_prediction(frame , self.face_feature_extractor , self.model , self.encoder).package()
                        cv2.imshow(str(self.source[i]) ,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                        buffer_labels[str(self.source[i])].append(label)    
                        #ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                        #frame = buffer.tobytes()
                        #yield (b'--frame\r\n'
                        #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        no_of_frame += 1
                        if(no_of_frame % 10 == 0):
                            for i in self.source:
                                final_labels.append(max(buffer_labels[str(i)]))
                                buffer_labels[str(i)] = []

                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    cv2.destroyAllWindows()
                    for i in camera:
                        i.release()
                    print(final_labels)
                    break
        except Exception as e:
            print(e)






import os
path = os.getcwd()
source = [0]
Camera(path , source).package()

"""
file = open("/Users/santoshsaxena/Desktop/Attendence_system/template/demo.html","w")
file.write("\n<html>")
for i in os.listdir("/Users/santoshsaxena/Desktop/Attendence_system/Trained_images/Female/"):
    file.write("\n<img src = '/Trained_images/Female/" +str(i) + "' alt = error> </img>")
file.write("\n</html>")
file.close()"""

