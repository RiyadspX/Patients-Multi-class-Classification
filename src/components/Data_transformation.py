import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# 1. Create a data transfo config to configure path 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")
    
# Creating Data transfo class 
class DataTransformation:
    """

    """
# Initialise the path config within the init
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates and returns a data transformation pipeline.

    This function defines separate preprocessing pipelines for numerical 
    and categorical features. These pipelines handle tasks such as missing 
    value imputation, scaling, and encoding, and are combined into a single 
    `ColumnTransformer` object for application on a dataset.

    Returns
    -------
    ColumnTransformer
        A ColumnTransformer object that applies:
        - A pipeline for numerical columns: Imputation (median strategy) and scaling (StandardScaler).
        - A pipeline for categorical columns: Imputation (most frequent strategy), encoding (OneHotEncoder), 
          and scaling (StandardScaler with `mean=False`).

    Raises
    ------
    CustomException
        If an error occurs during the creation of the preprocessing pipeline.

    Notes
    -----
    - The numerical columns processed by the pipeline are : Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'
    - The categorical columns processed by the pipeline are: Outcome
        

    Examples
    --------
    >>> transformer = get_data_transformer_object()
    >>> type(transformer)
    <class 'sklearn.compose._column_transformer.ColumnTransformer'>
        
        '''
        try:
            numerical_columns = ['N_Days', 'Bilirubin', 'Cholesterol', 
                                 'Albumin', 'Copper', 'Alk_Phos',
                                'SGOT', 'Tryglicerides', 'Platelets', 
                                'Prothrombin', 'Stage']
            categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
           

            num_pipeline = Pipeline(
                steps=[
                ("imputer", IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state = 0)),
                ("scaler", RobustScaler())

                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",    num_pipeline,   numerical_columns),
                ("cat_pipelines",   cat_pipeline,  categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Read the train. test that will come from data_ingestion
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Creating the preprocessor
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            # Preparing the X_train Y_train X_test Y_test
            target_column_name = "Outcome"
            numerical_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

            # Train    
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Test 
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit the preprocessor and transform train and test data 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #  concatenate arrays features and target along the second axis (i.e., columns)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            
            logging.info(f"Saving the preprocessing object.")
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )
            logging.info(f"Saved ! preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)







