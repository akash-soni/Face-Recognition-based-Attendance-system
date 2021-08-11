"""
Image_Processing Module

Image_Processing is a module which is supposed to fetch  the images from good images in a list called images  and fetch the file name 
in a list called names.Extracting face from images available in a list using mtcnn and override in the 
same list.Extract name from the file name available in the name list and override names in the same 
name list.

In this module there is a function called facenetmodel,in this function all the 
faces available in the image list will get in FaceNet model and the output 
will be stored in the same list.At the end all the list combined in a data 
frame.

Input : Path : path of current working directory
Output :  Dataframe containing the 512 features  of face and it's corresponding names

Approved on : 2/6/2021 
No of Revision : 1

"""

#Module Declaration#
#------------------#
import os
import cv2
import sys
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet
import re
#------------------------------------#

# This code  fetches the images from good images in a list called images 
# and fetch the file name in a list called names.

class Image_Processing:
    def __init__(self ,logger ,path):
        """
        :param image:images path
        """
        try:
            self.logger = logger
            self.logger.add_in_logs("NAM","Image preprocessing")
            self.logger.add_in_logs("BEG","Image Preprocessing module initialized")
            self.logger.add_in_logs("CHK","Image preprocessing constructor initialized")
        
            self.path = path
            self.faces = [] #list to store faces
            self.names = [] #list to store names
            self.features = [] #list to store facenet features
            self.facenet_model = FaceNet() #loading the facenet model
            self.logger.add_in_logs("inf","facenet model is loaded")
            self.mtcnn_model = MTCNN()        #loading the mtcnn model
            self.logger.add_in_logs("inf","mtcnn model is loaded")
            self.logger.add_in_logs("pas","Image preprocessing constructor completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Image preprocessing in initialization")
            self.logger.add_in_logs("LIN",'Line no is {}'.format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",e)

        
    def face_extractor(self):
        """
        This method extracts the faces using MTCNN 
        
        Input : path of current working directory
        Output : Image list which contains all the faces and name list which contains names 
        of the person.
        
        :param images: path of images
        :return: array
        """

        #intializing faces list
        try:

            self.logger.add_in_logs("chk","Face Extractor method initialized") 
            self.logger.add_in_logs("inf","Extracting face from images")
            #iterating through each image
            for image in os.listdir(self.path + '/Good_images'):
                self.logger.add_in_logs("inf", str(image) + " is selected")
                imagepath = os.path.join(self.path+'/Good_images/' , image)
                #reading the image
                img = cv2.imread(imagepath) 
                #converting image to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #extracting face from image
                result = self.mtcnn_model.detect_faces(img) 
                #getting dimensions from mtcnn result and
                #storing the cordinates
                x1, y1, width, height = result[0]["box"]
                x2, y2 = x1 + width, y1 + height
                #cropping the image using the coordinates
                img = img[y1:y2, x1:x2]
                #appending the face extracted to a list
                self.faces.append(img)  
                #appending it's name to a list
                image = image[:-4]
                image = re.sub('[0-9-]', '', image)
                self.names.append(image)
                self.logger.add_in_logs("inf", str(image) + " face is extracted")   
            self.logger.add_in_logs("pas","Faces Extracted method completed")
            #returning the faces and names
        #Exception Handling
        except Exception as e:
            self.logger.add_in_logs("ERR","Image preprocessing in Face Extraction")
            self.logger.add_in_logs("LIN",'Line no is {}'.format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def facenetmodel(self):
        """
        This method extracts 512 features from each face.
        
        input:face arrays from mtcnn
        output:returns the dataframe containing the 512 features 
        of face and it's corresponding names
        
        :param faces:array containing the faces
        """
        try:
            self.logger.add_in_logs("CHK","facenet model method initialized")
            self.logger.add_in_logs("inf","Processing faces into 512 facial features")
        
            #iterating through each face
            for face in self.faces:
                #facenet model accepts image of shape(160,160,3). so reshaping array 
                face = cv2.resize(face, (160, 160))             
                face = np.reshape(face, (1, 160, 160, 3))
                #FaceNet has method called embeddings which extracts 512 features from face.
                embeddings = self.facenet_model.embeddings(face) 
                #adding the extracted features to a list
                self.features.append([embeddings])
                #self.logger.add_in_logs("inf",str(face) + " face is processed")

            del self.faces  # Deleting previous face list              
            #reshaping the arrays in features.this makes easier to add  to a dataframe
            self.features = [i.reshape(-1).tolist() for i in np.array(self.features,dtype=object)[:, 0]] 
            #creating the dataframe of features
            df = pd.DataFrame(self.features)
            #adding name column to a dataframe
            df['Names'] = np.array(self.names,dtype=object) 
            self.logger.add_in_logs("inf","Features Extracted Successfully")
            #Exporting the dataframe
            self.logger.add_in_logs("inf","Exporting file to csv")
            df.to_csv(self.path + '/facenet.csv' , index = False)
            self.logger.add_in_logs("pas","facenet model method completed")

        #Exception Handling
        except Exception as e: 
            self.logger.add_in_logs("err","Image preprocessing in Facenet model")
            self.logger.add_in_logs("lin",'Line no is {}'.format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("typ",str(e))
    
    def package(self):
        try:
            self.face_extractor()
            self.facenetmodel()
            self.logger.add_in_logs("end","Image preprocessing module completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Image preprocessing in Package")
            self.logger.add_in_logs("Lin",'Line no is {}'.format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))
            

