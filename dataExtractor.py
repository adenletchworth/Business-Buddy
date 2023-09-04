import sqlite3 as sq3
import pandas as pd

def createDataBase():
    connection = sq3.connect('businessInsight.db')
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customerReviews
            ID INTEGER PRIMARY KEY,
            review_text TEXT,
            time TIMESTAMP,
            rating INTEGER,
            name TEXT,
            brand TEXT,
            category TEXT
    ''')

    connection.commit()
    cursor.close()
    connection.close()

class dataTransform:
    def __init__(self, dataBase):
        self.dataBase = dataBase
        self.connection = sq3.connect(self.dataBase)

    def addDataFrame(self, dataFrame, table_name):
        dataFrame.to_sql(table_name, self.connection, index=False, if_exists='append')

    def close(self):
        self.connection.close()
    

        

        



