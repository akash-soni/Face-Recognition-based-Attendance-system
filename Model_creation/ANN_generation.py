import sys
import os
import shutil
import pickle
from keras.callbacks import EarlyStopping
from keras.layers import Dense , Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import mysql.connector
from keras.models import Model , Sequential
import pandas as pd
import numpy as np

class ANN_generation:

    def __init__(self, path , logger , db_name , table_name):
        try:
            self.logger = logger
            self.path  = path
            self.db_name = db_name
            self.table_name = table_name
            self.logger.add_in_logs("NAM","Model creation")
            self.logger.add_in_logs("Beg","Model creation module initialized")

        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in initialization")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
    
    def loading_dataset(self):
        try:
            self.logger.add_in_logs("chk","loading dataset method initialized")
            self.logger.add_in_logs("chk","connecting database initialized")
            self.db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                password = "Santoshkyn14@",
                database = self.db_name
            )
            if(self.db):
                self.logger.add_in_logs("pas","database connected successfully")
            else:
                self.logger.add_in_logs("inf","database connection failed")
            self.cursor = self.db.cursor(buffered=True)
            self.cursor.execute("select * from  " + str(self.table_name))
            self.logger.add_in_logs("inf","extracting dataset from database")
            self.df = pd.DataFrame(self.cursor.fetchall())
            self.db.commit()
            self.logger.add_in_logs("inf","splitting the data into X and y")
            self.X = self.df.drop([512] , axis = 1)
            self.y = self.df.iloc[:,-1]
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            self.logger.add_in_logs("pas","loading dataset method completed")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in loading dataset")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
    
    def processing_labels(self):
        try:
            self.logger.add_in_logs("chk","processing labels module initialized")
            self.logger.add_in_logs("chk","Label Encoding initialized")
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(self.y)
            self.logger.add_in_logs("pas","Label Encoded completed")
            self.logger.add_in_logs("chk","One hot encoding initialized")
            self.no_of_labels = len(pd.Categorical(y_encoded).categories)
            self.y_onehot = np_utils.to_categorical(y_encoded , self.no_of_labels)
            self.logger.add_in_logs("inf","saving encoder file")
            pickle.dump(encoder, open(self.path + "/Model_creation/encoder.pickle","wb"))
            self.logger.add_in_logs("pas","one hot encoding Encoding completed")
            self.logger.add_in_logs("pas","processing labels module completed")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in processing labels")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))

    def selecting_no_of_layers(self):
        try:
            self.logger.add_in_logs("chk","selecting number of layers method initialized")
            accuracy = []
            layers = []
            max_no_of_layer = 10
            for i in range(1,max_no_of_layer):
                model = Sequential()
                for j in range(1,i+1):
                    if( j==1 and j== i):
                        model.add(Dense(self.no_of_labels * ((i+1) - j) , input_dim = 512 , activation = "sigmoid"))
                    elif(j == 1):
                        model.add(Dense(self.no_of_labels * ((i+1) - j) , input_dim = 512 , activation = "relu"))
                    elif(j == i):
                        model.add(Dense(self.no_of_labels * ((i+1) - j) , activation = "sigmoid" ))
                    else:
                        model.add(Dense(self.no_of_labels * ((i+1) - j) , activation = "relu" ))
        
                model.compile(loss="categorical_crossentropy" , optimizer="adam",metrics=["accuracy"])
                
                history = model.fit(self.X , self.y_onehot , epochs = 1)
                accuracy.append(history.history["accuracy"][0])
                layers.append(i)
            
            for i in range(len(accuracy)):
                if(accuracy[i] == max(accuracy)):
                    self.layers = layers[i]
            print("selecting layers")
            print(layers)
            print(accuracy)
            self.logger.add_in_logs("inf",str(self.layers) + " number of ANN layers is selected for training")
            self.logger.add_in_logs("pas","selecting number of layers method is completed")

        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in selecting number of layers")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
    
    def selecting_epochs(self):
        try:
            self.logger.add_in_logs("chk","selecting epcohs method initialized")
            max_no_of_epochs = 100
            model = Sequential()
            for j in range(1, self.layers+1):
                if( j==1 and j== self.layers):
                        model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , input_dim = 512 , activation = "sigmoid"))
                elif(j == 1):
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , input_dim = 512 , activation = "relu"))
                elif(j == self.layers):
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , activation = "sigmoid" ))
                else:
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , activation = "relu" ))
            model.compile(loss="categorical_crossentropy" , optimizer="adam",metrics=["accuracy"])
            history = model.fit(self.X , self.y_onehot , epochs = max_no_of_epochs)
            accuracy = history.history["accuracy"]

            accuracy_not_got = True

            for i in range(len(accuracy)):
                if(accuracy[i] >= 0.95):
                    self.epoch = i+1
                    accuracy_not_got = False
                    break
            self.logger.add_in_logs("inf", "accuracy not got is set " + str(accuracy_not_got))
            
            if(accuracy_not_got):
                self.logger.add_in_logs("inf","selecting max epochs as number of epochs")
                self.epoch = max_no_of_epochs

            self.logger.add_in_logs("inf",str(self.epoch) + " number of epochs is selected for training")
            self.logger.add_in_logs("inf",str(history.history["accuracy"][self.epoch]) + " is the accuracy at this epoch")
            self.logger.add_in_logs("pas","number of epochs method completed")
            del model
        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in selecting epochs")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))

    
    def training_ANN(self):
        try:
            self.logger.add_in_logs("chk","training ANN method initialized")
            model = Sequential()
            for j in range(1 , self.layers + 1):
                if( j==1 and j== self.layers):
                        model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , input_dim = 512 , activation = "sigmoid"))
                elif(j == 1):
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , input_dim = 512 , activation = "relu"))
                elif(j == self.layers):
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , activation = "sigmoid" ))
                else:
                    model.add(Dense(self.no_of_labels * ((self.layers + 1) - j) , activation = "relu" ))
            model.compile(loss="categorical_crossentropy" , optimizer="adam",metrics=["accuracy"])
            self.logger.add_in_logs("inf","Training is starting")
            history = model.fit(self.X , self.y_onehot , epochs = self.epoch)
            self.logger.add_in_logs("inf", str(history.history["accuracy"][-1]) + " is the accuracy of the model")
            if(os.path.isdir(self.path + "/Model_creation/ANN_model.model")):
                shutil.rmtree(self.path + "/Model_creation/ANN_model.model")
            model.save(self.path + "/Model_creation/ANN_model.model")
            self.logger.add_in_logs("pas","Training method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in training ANN")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
    
    def package(self):
        try:
            self.loading_dataset()
            self.processing_labels()
            self.selecting_no_of_layers()
            self.selecting_epochs()
            self.training_ANN()
            self.logger.add_in_logs("end","Model creation module completed")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "ANN generation in package")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
