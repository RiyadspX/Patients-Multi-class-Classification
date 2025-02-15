import pandas as pd 
#
from src.components.Data_transformation import DataTransformation
from src.exception import CustomException




if __name__ == '__main__':

    datatransfo = DataTransformation()
    X_sub = datatransfo.transform_test_data(data_path='Datasets/test.csv')
    print(X_sub.shape)
    

