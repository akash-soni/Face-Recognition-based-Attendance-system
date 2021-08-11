import cv2
import sys
import pickle
import cvlib as cv
from cvlib import face_detection
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet


class Frame_prediction:
    def __init__(self , frame ,face_feature_extractor , model , encoder):
        self.frame = frame
        self.model = model
        self.face_feature_extractor = face_feature_extractor
        self.encoder = encoder
    
    def package(self):
        try:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.frame = cv2.flip(self.frame , 1)
            self.frame = cv2.cvtColor(self.frame , cv2.COLOR_BGR2RGB)
            result = cv.detect_face(self.frame)
            timer = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            answer = "No_one"

            for box , confidence in zip(result[0] , result[1]):
                if(confidence > 0.8):

                    x1 , y1  = box[0] , box[1]
                    x2 , y2 = box[2] , box[3]
                    img = self.frame[y1:y2 , x1:x2]

                    img = cv2.resize(img , (160,160))
                    img = np.reshape(img , (1,160,160,3))

                    face_feature = self.face_feature_extractor.embeddings(img)
                    prediction = self.model.predict(face_feature)[0]
                    print("vector",prediction)
                    if(max(prediction) > 0.95):
                        accuracy = round(max(prediction)*100,2)
                        prediction = np.argmax(prediction)
                        print("max",prediction)
                        label = self.encoder.inverse_transform([prediction])[0]
                        answer = label
                        label = "Name : {0} , Accuracy : {1}%".format(label, accuracy)
                        fps_label = "FPS : {0}".format(fps)
                        cv2.rectangle(self.frame, (x1,y1), (x2,y2), (0,255,0), 3)
                        cv2.putText(self.frame, label, (x1 - 50,y1 - 10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 2)
                        cv2.putText(self.frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

                    else:
                        accuracy = round((1 - max(prediction))*100,2)
                        label = "Unknown"
                        answer = label
                        label = "Name : {0} , Accuracy : {1}%".format(label , accuracy)
                        fps_label = "FPS : {0}".format(fps)
                        cv2.rectangle(self.frame, (x1,y1), (x2,y2), (255,0,0), 3)
                        cv2.putText(self.frame, label, (x1 - 50,y1 - 10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
                        cv2.putText(self.frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
                else:
                    cv2.putText(self.frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
                    continue

            return [self.frame , answer]


            #frame = self.frame
            #ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            #frame = buffer.tobytes()
            #yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




        except Exception as e:
            print(e)
            return self.frame
