from logging import log
import os
from Logger.logger import Logger
from Validation.Image_Validation import ImageValidate
from Validation.Name_Validation import Name_Validation
from Preprocessing.Image_Processing import Image_Processing
from Preprocessing.Database import Database
from Model_creation.ModelTraining import Model_creation


class Training:
    def package(self):
        try:
            path = os.getcwd()
            logger = Logger(path)
            im = ImageValidate(logger, path).package()
            val = Name_Validation(logger, path).package()
            pro = Image_Processing(logger, path).package()
            self.database_name = "Ineuron"
            self.table_name = "candidate_list"
            db = Database(path , logger , self.database_name , self.table_name).package()
            mo = Model_creation(logger, path , self.database_name , self.table_name).package()

        except Exception as e:
            print(e)

t = Training().package()
