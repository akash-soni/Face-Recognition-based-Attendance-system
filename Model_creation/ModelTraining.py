"""
Model Training module :


Working :
Their should be multiple method for Model creation. This methods are going to
be in module called Model Creation. Their will be Data preprocessing method
which will give a data frame in the form of self.df. Hence , One method will
select how many layers should be their in a ANN. One method will select how
many epochs are required for accuracy more than 95%. And last method will
select these parameters and form a final model that will be saved in the
directory. The name of the model file will be DNN_model.model .

Input : fetching 512 featured preprocessed data of images
Output : Model file in current working directory.

Approved on :
No of Revision :

"""

#Module Declaration#
#------------------#
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch, Hyperband
from keras.utils import np_utils
from sklearn import preprocessing
import pandas as pd
import sys
import numpy as np
import os
import shutil
import mysql.connector
import pickle
#------------------#

class Model_creation:
    def __init__(self,logger ,path , db_name , table_name):
        """
        This is initialization function. This function is constructor.
        The function will initialize path variable and will load dataset for traning

        Input : path
        Output : All the necessary variable assignment and load dataset for training

        """
        try:
            self.logger = logger
            self.logger.add_in_logs("Nam","model creation")
            self.logger.add_in_logs("BEG","Model creation module initialized")
            self.logger.add_in_logs("chk","model creation constructor initialized")
            self.path = path
            self.db_name = db_name
            self.table_name = table_name
            self.logger.add_in_logs("pas","model creation constructor completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Model creation in Initialization")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def loading_dataset(self):
        try:
            self.logger.add_in_logs("chk","loading dataset method initialized")
            self.logger.add_in_logs("chk","connecting database initialized")
            self.db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                password = "mysql",
                database = self.db_name
            )
            if(self.db):
                self.logger.add_in_logs("pas","database connected successfully")
            else:
                self.logger.add_in_logs("inf","database connection failed")
            self.cursor = self.db.cursor(buffered=True)
            self.cursor.execute("select * from  " + str(self.table_name))
            self.logger.add_in_logs("inf","extracting dataset from database")
            self.df1 = pd.DataFrame(self.cursor.fetchall())
            #self.df1 = self.df1.append([self.df1] * 35, ignore_index=False)
            self.db.commit()
            self.logger.add_in_logs("pas","loading dataset method completed")    
        except Exception as e:
            self.logger.add_in_logs("ERR" , "model creation in loading dataset")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))


    def remove_previous(self):
        """
        In case if the model is required to be retrained then this method
        will remove all the previously trained model and training trials

        Input : N/A
        Output : Removes previously trained model and training trials directories .
        :return:
        """
        try:
            self.logger.add_in_logs("chk","remove previous method intialized")
            # remove previous trials directory
            if os.path.isdir(self.path + '/' + 'test_dir/'):
                shutil.rmtree(self.path + '/' + 'test_dir/')
            self.logger.add_in_logs("inf","previous trials removed")

            # remove previously trained model
            if os.path.isdir(self.path + '/' + 'ANN_model/'):
                shutil.rmtree(self.path + '/' + 'ANN_model/')
            self.logger.add_in_logs("inf","previous trials removed")

            self.logger.add_in_logs("pas","remove_previous method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Model creation in remove previous")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def data_preprocesing(self):
        """
        data_preprocessing() method will seperate data and labels in X and y respectively.
        On labels it will then perform label encoding followed by One Hot Encoding.
        The function will also set Output dimensions for model.

        Input: preprocessed dataframe
        Output: X features and y labels as one hot encoded features

        :return:
        """
        try:
            self.logger.add_in_logs("chk","data preprocessing method initialized")

            self.X = np.array(self.df1.iloc[:, :-1])  # independent features
            self.y = np.array(self.df1.iloc[:, -1])  # dependent features
            self.logger.add_in_logs("inf","training features and labels initialized successfully")

            # label endcoding
            lbl_encoder = preprocessing.LabelEncoder()
            self.y_encoded = lbl_encoder.fit_transform(self.y)
            self.logger.add_in_logs("inf","Label encoding successful")


            # Onehot encoding
            self.total_labels = len(pd.Categorical(self.y_encoded))
            self.y_onehot_encoded = np_utils.to_categorical(self.y_encoded, self.total_labels)
            self.logger.add_in_logs("inf","One Hot encoding successful")

            output = open('encoder.pkl', 'wb')
            pickle.dump(lbl_encoder, output)
            output.close()

            # Output dimensions
            self.ouput_dimensions = len(self.y)

            self.logger.add_in_logs("pas","data preprocessing method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Model creation in data preprocessing")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def layers_processing(self):
        """
        layer_preprocessing() is responsible for tuning in best no. of layers
        and also performs hyperparameter tuning for learning rates

        Input : 512 featured data and One hot encoded labels
        Output : best number of layers and learning rate in variable best_hps
        :return: best_hps
        """
        try:
            self.logger.add_in_logs("chk","layer processing method initialized")

            # building a model to get best number of layers
            def build_model(hp):
                try:
                    self.model = keras.Sequential()
                    for i in range(hp.Int('num_layers', 2, 10)):
                        self.model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=256, max_value=512),
                                                activation='relu'))
                    self.model.add(layers.Dense(self.ouput_dimensions, activation='softmax'))
                    self.model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                                       loss='categorical_crossentropy',
                                       metrics=['accuracy'])
                    return self.model
                except Exception as e:
                    self.logger.add_in_logs("ERR","Problem in tuning")
                    self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
                    self.logger.add_in_logs("TYP",str(e))

            # initialize the tuner
            self.tuner = RandomSearch(
                                build_model,
                                objective='accuracy',
                                # max_epochs=10,
                                max_trials=5,
                                executions_per_trial=3,
                                directory=self.path+'/'+'test_dir')
                                # hyperband_iterations=1)

            # self.tuner = Hyperband(
            #                     build_model,
            #                     objective='accuracy',
            #                     max_epochs=10,
            #                     # max_trials=5,
            #                     executions_per_trial=3,
            #                     directory='test_dir',
            #                     project_name='trial')

            self.tuner.search_space_summary()

            # Run the tuner on X data and y labels for given no. of epochs
            self.tuner.search(x=self.X, y=self.y_onehot_encoded, epochs=10, verbose=1)

            # Results of top 10 trials are arranged from best to worst so choose the result of first trial
            self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
            self.logger.add_in_logs("inf","best number layers "+ str(self.best_hps['num_layers']))
            self.logger.add_in_logs("inf","best learning rate "+ str(self.best_hps['learning_rate']))
            print("best number layers ", self.best_hps['num_layers'])
            print("best learning rate ", self.best_hps['learning_rate'])
            self.logger.add_in_logs("pas","layer processing method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Model creation in layer processing")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def epochs_processing(self):
        """
        epochs_preprocessing() will return best epoch value for which the model should run

        Input: 512 featured data, One hot encoded labels and best_hps
        Output: best epoch value
        :return: best_epoch
        """
        try:
            max_no_of_epochs = 100
            self.logger.add_in_logs("chk","epochs processing method initialized")
            # finding best number of epochs on the data
            # Build the model with the optimal hyperparameters and train it on the data for 30 epochs
            self.logger.add_in_logs("inf","start finding best epoch")
            self.model = self.tuner.hypermodel.build(self.best_hps)
            self.history = self.model.fit(self.X, self.y_onehot_encoded, epochs=max_no_of_epochs)

            self.loss = self.history.history['loss']
            self.accuracy = self.history.history['accuracy']
            #self.val_acc_per_epoch = self.history.history['accuracy']
            #self.best_epoch = self.val_acc_per_epoch.index(max(self.val_acc_per_epoch)) + 1  # best epochs value

            accuracy_not_got = True

            for i in range(len(self.accuracy)):
                if (self.accuracy[i] >= 0.95 and self.loss[i] <= 0.3):
                    self.best_epoch = i + 1
                    accuracy_not_got = False
                    break
            self.logger.add_in_logs("inf", "accuracy not got is set " + str(accuracy_not_got))

            if (accuracy_not_got):
                self.logger.add_in_logs("inf", "selecting max epochs as number of epochs")
                self.best_epoch = max_no_of_epochs

            self.logger.add_in_logs("inf",'Best epoch: ' + str(self.best_epoch))
            print("inf", 'Best epoch: ' + str(self.best_epoch))
            self.logger.add_in_logs("inf","best epoch found,")
            self.logger.add_in_logs("pas","epochs processing method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR","Model creation in epochs processing")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def final_processing(self):
        """
        final_processing() will train final model using the best number of layers,
        learning_rate and epochs obtained from previous processing.

        Input : 512 featured data, One hot encoded labels, best_hps, best_epoch
        Output : DNN_model

        :return:
        """
        try:
            self.logger.add_in_logs("chk","final processing method initialized")
            print("build model with best hyperparameters")
            # final model training with best hyperparmaeters and epochs
            hypermodel = self.tuner.hypermodel.build(self.best_hps)
            # Retrain the model
            self.logger.add_in_logs("inf","Final model training started")
            print("use best epochs")
            hypermodel.fit(self.X, self.y_onehot_encoded, epochs=self.best_epoch)
            self.logger.add_in_logs("inf","Final Model training completed")
            print("Final Model training completed")
            # save the model to disk
            hypermodel.save(self.path + "/ANN_model")
            self.logger.add_in_logs("inf","Model saved")
            self.logger.add_in_logs("pas","final processing method completed")
        except Exception as e:
            self.logger.add_in_logs("ERR","Training problem ")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))

    def package(self):
        try:
            self.loading_dataset()
            self.remove_previous()
            self.data_preprocesing()
            self.layers_processing()
            self.epochs_processing()
            self.final_processing()
            self.logger.add_in_logs("end","model creation module completed")
        except Exception as e:
            self.logger.add_in_logs("ERR","Model Training in package")
            self.logger.add_in_logs("LIN","Line no is " + str(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP",str(e))
