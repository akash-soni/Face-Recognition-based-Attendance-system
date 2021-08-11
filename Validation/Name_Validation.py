''' 
Name_Validation Module
NameValidation is a module which is supposed to fatch images from good_image directory and move into bad_directories.
If image availbele in good_image directory satisfies all the validation condition then that image will be there in the good_directory else move into bad directory.
The ground truth of validation conditions should be fetched from source of truth called sot.json

Input : Fatch images from good_image directory
Output : Segregation of image in good_images and bad_iamges

Approved on : 2/6/2021
No of Revison : 1

'''

# Module Decleartion
#--------------------#
import os
import sys
import json
import shutil
# ---------------------#


class Name_Validation:
    def __init__(self,logger,path):
        '''
        This is initialization function. This function is constructor.
        All the necessay variables are assigned in this function

        Input : path
        Output : All the necessary variable asignment
            
        '''
        try:
            self.logger = logger
            self.logger.add_in_logs("NAM","Name validation")
            self.logger.add_in_logs("BEG","Name validation module initialized")
            self.logger.add_in_logs("CHK","Name Validation constructor intialzed")

            # Initialize the directory path variable
            self.path = path

            # Checking availability of good_image directory
            
            if not os.path.isdir(self.path + '/Good_images'):
                self.logger.add_in_logs("inf","Good_images directory  not found")
                raise("Good images directory not found")
            else:
                self.logger.add_in_logs("inf","Good_Images Directory  found")

            # Load JSON source of truth(SOT) file
            self.sot=json.load(open(self.path+'/Source_of_truth/SOT.json',"r"))
            self.logger.add_in_logs("inf","SOT loaded successfully")
            
            self.logger.add_in_logs("pas","Name validation constructor completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Error in Name Validation Intialization")
            self.logger.add_in_logs("LIN",f"Line No is {sys.exc_info()[-1].tb_lineno}")
            self.logger.add_in_logs("TYP",str(e))
    
    def person_name_validation(self):
        '''
        This method is responsible for validation and moving  image form good_image directory
        to bad_image directory if validation condition doesn't satisfies.

        Input: N/A
        Output: Separate all the images to respective directory   
               
        '''
        try:
            self.logger.add_in_logs("chk","Person Name validation method initialized")

            # listing all .jpg file name into self.jpg_file
            # finding the length of file name like (first_name and last_name) then its length is 2
            # check if directory containing image or not

            for file_name in os.listdir(self.path + "/Good_images"):        
                
                # split the name by spaces
                self.split_name=file_name.split('.')[0].split(' ')
                self.logger.add_in_logs("inf" , str(self.split_name) + " is selected")
                if(len(self.split_name) != len(self.sot['name'].split(' '))):
                    # Moving the file bad_directory
                    shutil.move(self.path + "/Good_images/" + str(file_name) , self.path + "/Bad_images")
                    self.logger.add_in_logs("inf",str(file_name) + " is not satisfying validation conditions")
                    self.logger.add_in_logs("inf",str(file_name) + " is moved to Bad_images directory")
                else:
                    self.logger.add_in_logs("inf",str(file_name) + " is satisfying validation conditions") 
                    self.logger.add_in_logs("inf",str(file_name) + " is remained to Good_images directory")

            self.logger.add_in_logs("pas","Person name validation method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Person Name Validation method error")
            self.logger.add_in_logs("LIN",f"Line No is{sys.exc_info()[-1].tb_lineno}")
            self.logger.add_in_logs("TYP",str(e))
        
    def package(self):
        try:
            self.person_name_validation()
            self.logger.add_in_logs("END","Name validation module Completed")
        except Exception as e:
            self.logger.add_in_logs("ERR","Name validation error in package")
            self.logger.add_in_logs("LIN",f"Error on line number{sys.exc_info()[-1].tb_lineno}")
            self.logger.add_in_logs("TYP",str(e))

