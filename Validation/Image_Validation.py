"""
Image_Validate Module

ImageValidate is a module which is supposed to fetch images from Training_Batch_Folder and move images
from Training_Batch_Folder into Good_images and Bad_images directories. if image available in Training_Ba-
tch_Folder satisfies all the validation conditions then that image will move to Good_images directory else
Bad_images directory. The ground truth of validation conditions should be fetch from source of truth called 
SOT.json. 

Input : Fetching images from Training_Batch_Folder
Output : Segregations of images in Good_images and Bad_images with respect to the validation conditions.

Approved on : 2/6/2021
No of Revision : 1

"""

#Module Declaration#
#------------------#
import os
import sys
import cv2
import json
import shutil
from mtcnn import MTCNN
#------------------------------------#

class ImageValidate:
    def __init__(self, logger ,path):
        """
        This is initialization function. This function is constructor.
        All the necessay variables are assigned in this function

        Input : path
        Output : All the necessary variable assignment.
        
        """
        try:

            self.logger = logger
            self.logger.add_in_logs("nam","Image Validation")
            self.logger.add_in_logs("BEG","Image Validation module initialized")
            self.logger.add_in_logs("CHK","Image validation constructor initialized")
            
            # initialize the directory paths variables
            self.path = path        

            # Checking availablity of Training_Batch_Folder
            if not os.path.isdir(self.path + '/'+'Training_Batch_Folder'):
                self.logger.add_in_logs("inf","Training_Batch_Folder not found ")
                raise("Training_Batch_Folder not found")
            else:
                self.logger.add_in_logs("inf","Training_Batch_Folder found")

            # Load JSON source of truth(SOT) file
            self.sot = json.load(open(self.path + '/Source_of_Truth/' + 'SOT.json' , "r"))
            self.logger.add_in_logs("inf","SOT loaded successfully ")

            # initalize MTCNN model for face detection
            self.detector = MTCNN()
            self.logger.add_in_logs("inf","MTCNN initialization successfully")
            
            self.logger.add_in_logs("pas","Image validation constructor completed")
        except Exception as e:
            self.logger.add_in_logs("ERR","Image Validation in Initialization")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("inf","TYP",str(e))


    def directory_generation(self):
        """
        Generating Good_images and Bad_images directory

        Input : N/A
        Output : Good_images and Bad_images directory generation in current working direcotry.

        """
        try:
            self.logger.add_in_logs("chk","directory generation method initialized")

            # generating Good_image directory
            self.logger.add_in_logs("chk","good_images directory generation initialized")
            if os.path.isdir(self.path + '/' + 'Good_images/'):
                shutil.rmtree(self.path + '/' + 'Good_images/')
                self.logger.add_in_logs("inf","Good_images directory already exists")
                self.logger.add_in_logs("inf","Deleting previously generating directory")
            os.mkdir(self.path + '/' + 'Good_images')  # directory created
            self.logger.add_in_logs("inf","Good_images directory created")
            self.logger.add_in_logs("pas","good_images directory generation completed")

            self.logger.add_in_logs("chk","bad_images directory generation initialized")
            # generating Bad_image directory
            if os.path.isdir(self.path + '/' + 'Bad_images/'):
                shutil.rmtree(self.path + '/' + 'Bad_images/')
                self.logger.add_in_logs("inf","Bad_images directory already exists")
                self.logger.add_in_logs("inf","Deleting previously generating directory")
            os.mkdir(self.path + '/' + 'Bad_images')  # directory created
            self.logger.add_in_logs("inf","Bad_images directory created")
            self.logger.add_in_logs("pas","Bad_images directory completed")

            self.logger.add_in_logs("pas","directory generation method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Image Validation in Initialization")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def ImTypeValidation(self):
        """
        This method is fetching all the images from Training_Batch_Folder and sending all the path
        to the method called face_validate for validation.

        Input : N/A
        Output : Images in Good_images and Bad_images directories.

        """
        try:
            self.logger.add_in_logs("chk","Image validation method initialized")
            
            # get all the files inside the current directory and feed it to validation function
            self.logger.add_in_logs("inf","Fetching all images from training_batch_folder")
            for img in os.listdir(self.path + "/Training_Batch_Folder"):
                self.logger.add_in_logs("inf",str(img) + " is selected for validation")
                if img.endswith(".jpg"):  
                    self.logger.add_in_logs("inf", str(img) + " is ending with .jpg")
                    self.face_validate(self.path + "/Training_Batch_Folder/" + str(img))  # call face_validation function to check faces in the image
                else:  # if file is not .jpg then copy it to bad_images directory
                    self.logger.add_in_logs("inf",str(img) + " is not ending with .jpg")
                    shutil.move(self.path +"/Training_Batch_Folder/" + str(img) , self.path+"/Bad_images")
                    self.logger.add_in_logs("inf","file format is not .jpg")
                    self.logger.add_in_logs("inf","Moved to Bad_images directory")

            self.logger.add_in_logs("pas","Image validation method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Image Validation in ImTypeValidation")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

        # function to validate face in the image
    def face_validate(self, path_to_image):
        
        """
        This method is responsible for validation and moving images from 
        Training_batch_Folder to Good_images and Bad_images directory.

        Input : path_to_image : path of a particular image
        Output : Images into Good_images and Bad_images directory. 

        """

        try:
            self.logger.add_in_logs("chk","face validation method initialized")

            image = cv2.imread(path_to_image, cv2.COLOR_BGR2RGB)  # Reading image
            if image is not None:
                detected_faces = self.detector.detect_faces(image)  # detect faces if image is read correctly
                if len(detected_faces) == self.sot["no_of_person_per_image"]:  # if number of face detected is 1
                    self.logger.add_in_logs("inf","one person per image validation condition satisfied")
                    if image.shape[2] == self.sot["Layers"]:  # if number of channels in image are 3
                        self.logger.add_in_logs("inf","3 channels validation condition satisfied")
                        if len(image.shape) == self.sot["Dimension"]:  # if shape of image is n*n*c then
                            self.logger.add_in_logs("inf","3 dimensions image validation condition satisfied")
                            # moving image to good_images folder
                            shutil.move(path_to_image, self.path + "/Good_images")
                            self.logger.add_in_logs("inf","satisfied all the validation conditions")
                            self.logger.add_in_logs("inf","moved image to Good_image directory")
                        else:
                            # writing rejected image to bad_images folder
                            shutil.move(path_to_image, self.path + "/Bad_images/")
                            self.logger.add_in_logs("inf","3 dimensions image validation condition failed")
                            self.logger.add_in_logs("inf","Moved to Bad_images directory")
                    else:
                        shutil.move(path_to_image, self.path + "/Bad_images")
                        self.logger.add_in_logs("inf","3 channels validation condition failed")
                        self.logger.add_in_logs("inf","Moved to Bad_images directory")
                else:
                    shutil.move(path_to_image, self.path + "/Bad_images")
                    self.logger.add_in_logs("inf","one person per image validation condition failed")
                    self.logger.add_in_logs("inf","Moved to Bad_images directory")
            else:
                shutil.move(path_to_image, self.path + "/Bad_images")
                self.logger.add_in_logs("inf","Image not found")
                self.logger.add_in_logs("inf","Moved to Bad_images directory")
            
            self.logger.add_in_logs("pas","image validation method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Image Validation in face_validate")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))


    def package(self):
        try:
            self.directory_generation()
            self.ImTypeValidation()
            self.logger.add_in_logs("END","Image validation module completed")
        except Exception as e:
            self.logger.add_in_logs("ERR","Image Validation in package")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",e)


