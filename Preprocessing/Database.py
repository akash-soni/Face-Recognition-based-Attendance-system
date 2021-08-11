
import os
import sys
import pandas as pd
import mysql.connector


class Database:

    def __init__(self ,path ,logger,db_name , table_name , begining_start = False):
        try:
            self.path = path
            self.logger = logger
            self.df = pd.read_csv(self.path + "/facenet.csv")
            self.db_name = db_name
            self.table_name = table_name
            self.begining_start = begining_start
            self.logger.add_in_logs("Nam","Database")
            self.logger.add_in_logs("beg","Database module initialization")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in initialization")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))


    def create_db(self):
        try:
            self.logger.add_in_logs("chk","Creating database method initialized")
            self.db = mysql.connector.connect(
                host = 'localhost',
                user = 'root',
                password = 'mysql'
                )
            if(self.db):
                self.logger.add_in_logs("inf","connection done successfully")
            else:
                self.logger.add_in_logs("inf","connectoion failed")
                raise("Connection failed")
            cursor = self.db.cursor(buffered=True)
            cursor.execute("show databases")
            db_already_exists = False

            for i in cursor:
                if(i[0] == self.db_name):
                    db_already_exists = True
                else:
                    continue

            if not db_already_exists:
                self.logger.add_in_logs("inf","Database does not exists")
                self.logger.add_in_logs("inf","Creating a new database")
                cursor.execute("create database " + str(self.db_name))
            else:
                if(self.begining_start):
                    self.logger.add_in_logs("inf","Begining start is true")
                    self.logger.add_in_logs("inf","deleting the previous data")
                    cursor.execute("drop database " + str(self.db_name))
                    cursor.execute("create database " + str(self.db_name))
                else:
                    self.logger.add_in_logs("inf","Database exists")
                    self.logger.add_in_logs("inf","Using previosusly existed data")

            self.logger.add_in_logs("pas","creating database method completed")
            self.db.commit()

        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in initialization")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))

    def create_table(self, mycursor ,table_name , columns , columns_type , num , features = [], primary_key = [], foreign_key = [], reference = []):
        """
        This is a function to generate a query for creating a table

        Input : basic inputs to generate queries
        Output : Table in database

        """
                
        try:
            if(columns == [] or columns_type == [] and mycursor == ""):
                raise("ERR","attributes are missing")
            else:
                string = "create table if not exists " + table_name + "("
                for i,j,k in zip(columns, columns_type, num):
                    if(j == "float"):
                        k = ""
                    else:
                        k = "("+str(k) +")"
                    string = string + i+" "+j  + str(k) 
                    if(features != []):
                        string = string + features.pop(0) + ","
                    else:
                        string = string + ","
                if(primary_key != []):
                    string = string + "primary key(" + primary_key.pop(0) + ")," 
                for i,j in zip(foreign_key, reference):
                    string = string + "foreign key (" + i + ") " + "references " + j+","
                string = string[0:len(string) - 1]
                string = string + ")"
                mycursor.execute(string)
        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in create table")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))


    def insert_into_table(self, mycursor,table_name=[], values=[]):
        """
        This is a function to generate a query for insertion of observation into database table 

        Input : observations to add in database
        Output : observation in a database table
        """
        
        try:
            if(table_name == [] or values == []):
                raise("parameters of table are missing")
            else:
                string = "insert into "+table_name 
                mycursor.execute("desc " + table_name)
                string = string + "("
                for i in mycursor:
                    string = string + i[0] +","
                string = string[0: len(string) - 1]
                string = string + ")"
                string = string + " values("
                for i in values:
                    if(type(i) == str):
                        string = string + "'"
                        string = string + "{}".format(i)
                        string = string + "',"
                    else:
                        string = string + "{}".format(i)
                        string = string + ","
            string = string[0:len(string)-1]
            string = string + ")"
            mycursor.execute(string)
        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in insert into table")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))

    def table_creation(self):
        try:
            self.logger.add_in_logs("chk","creating table method initialized")
            self.db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                password = "mysql",
                database = self.db_name                
            )
            if(self.db):
                self.logger.add_in_logs("inf","connection build successfully")
            else:
                self.logger.add_in_logs("inf","connection failed")

            self.logger.add_in_logs("inf","checking previosuly available table")

            self.cursor = self.db.cursor(buffered=True)
            self.cursor.execute("show tables")
            existing_table = False
            for i in self.cursor:
                if(i[0] == self.table_name):
                    existing_table = True
            
            column_name = []
            for i in range(len(self.df.columns) - 1):
                string = str(i)
                string = string + "_feature"
                column_name.append(string)
    
            column_type = ["float"]*len(column_name)

            column_type.append("varchar")
            column_name.append("Name")


            if not existing_table:
                self.logger.add_in_logs("inf","table does not exists")
                self.logger.add_in_logs("inf","creating a new table")
                self.create_table(
                    mycursor=self.cursor,
                    table_name=self.table_name,
                    columns =  column_name,
                    columns_type= column_type,
                    num = [100]*len(column_type)
                )
            else:
                self.logger.add_in_logs("inf","table exists")
                self.logger.add_in_logs("inf","using same table")
            
            self.db.commit()
            self.logger.add_in_logs("pas","checking table method completed")

        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in table creation")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))

    def insert_data_into_db(self):
        try:
            self.logger.add_in_logs("chk","Inserting data into database")
            self.logger.add_in_logs("inf","setting up database connection")
            self.db = mysql.connector.connect(
                host = "localhost",
                user = "root",
                password = "mysql",
                database = self.db_name
            )
            if(self.db):
                self.logger.add_in_logs("inf","database connected successfully")
            else:
                self.logger.add_in_logs("inf", "database connection failed")
            
            self.cursor = self.db.cursor(buffered=True)
            self.logger.add_in_logs("inf","inserting data into database")
            for i in range(len(self.df)):
                self.insert_into_table(mycursor=self.cursor , table_name=self.table_name , values=list(self.df.iloc[i]))
            self.logger.add_in_logs("pas","data inserted into database")
            self.db.commit()
            self.logger.add_in_logs("inf","Removing the temporary dataframe")
            os.remove(self.path + "/facenet.csv")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in inserting database")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))
    

    def package(self):
        try:
            self.create_db()
            self.table_creation()
            self.insert_data_into_db()
            self.logger.add_in_logs("end","Database module completed")
        except Exception as e:
            self.logger.add_in_logs("ERR" , "database in package")
            self.logger.add_in_logs("LIN" , "Error on line number : {}".format(sys.exc_info()[-1].tb_lineno))
            self.logger.add_in_logs("TYP" , str(e))




            


