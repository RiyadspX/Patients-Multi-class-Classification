import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
import os
from sklearn.compose import ColumnTransformer
import pandas as pd


def encoding_mixed(X_train_cat_imputed, X_test_cat_imputed, onehot_cols, ordinal_cols):
    # Define the transformers for one-hot and ordinal encoding
    transformers = [
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
    ]

    # Create the ColumnTransformer
    column_transformer = ColumnTransformer(transformers, remainder='passthrough')

    # Fit the transformer on the training set and transform both datasets
    X_train_encoded = column_transformer.fit_transform(X_train_cat_imputed)
    X_test_encoded = column_transformer.transform(X_test_cat_imputed)

    # Get feature names for one-hot encoded columns
    onehot_feature_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(onehot_cols)

    # Combine the one-hot and ordinal column names
    all_columns = list(onehot_feature_names) + ordinal_cols + [
        col for col in X_train_cat_imputed.columns if col not in onehot_cols + ordinal_cols
    ]

    # Convert to DataFrame with proper column names
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=all_columns, index=X_train_cat_imputed.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=all_columns, index=X_test_cat_imputed.index)

    return X_train_encoded, X_test_encoded

def target_encoding(Y):
    code = {'C':1, 'D':0, 'CL':2}
    Y = Y.map(code)
    return Y

def imputing_numerical(X_train_num, X_test_num):
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state = 0)
        imputer.fit(X_train_num)
        X_train_num_imputed = imputer.transform(X_train_num)
        X_test_num_imputed = imputer.transform(X_test_num)

        X_train_num_imputed = pd.DataFrame(X_train_num_imputed)
        X_train_num_imputed.columns = X_train_num.columns
        X_test_num_imputed = pd.DataFrame(X_test_num_imputed)
        X_test_num_imputed.columns = X_test_num.columns
        return X_train_num_imputed, X_test_num_imputed

def imputing_categorical(df_fill):
    for col in df_fill.columns:
        if df_fill[col].isna().any():
            df_fill.loc[:, col] = df_fill[col].fillna(df_fill[col].mode()[0])
    return df_fill


def scaling(X_train, X_test):
    X_train_tf = X_train.copy()
    X_test_tf = X_test.copy()
    numerical_features = [col for col in X_train.select_dtypes('float').columns if col != 'Stage']
    #only scale numeric varaibles in this case rather than the dummy variables for categories 
    rob = RobustScaler()
    X_train_tf.loc[:, numerical_features] = rob.fit_transform(X_train_tf.loc[:, numerical_features])
    X_test_tf.loc[:, numerical_features] = rob.transform(X_test_tf.loc[:, numerical_features])
    return X_train_tf, X_test_tf

# 1. Create a data transfo config to configure path 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")
    
# Creating Data transfo class 
class DataTransformation:
# Initialise the path config within the init
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try : 

            # Read the train test that will come from data_ingestion
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            # Preparing the X_train Y_train X_test Y_test
            target_column_name = "Status"
            numerical_columns = ['N_Days', 'Bilirubin', 'Cholesterol', 
                                'Albumin', 'Copper', 'Alk_Phos',
                                'SGOT', 'Tryglicerides', 'Platelets', 
                                'Prothrombin', 'Stage']
            categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
            columns_delete = ['id', 'Age']

            # Training vectors    
            X_train = train_df.drop(columns= columns_delete + [target_column_name], axis=1)
            Y_train = train_df[target_column_name]

            # Test vectors 
            X_test = test_df.drop(columns=columns_delete + [target_column_name], axis=1)
            Y_test = test_df[target_column_name]
            logging.info("Split the trainning and testing vectors (X, Y) completed")

            # SPLITTING Categorical and numerical 
            # Train 
            X_train_num = X_train[numerical_columns]
            X_train_cat = X_train[categorical_columns]
            # Test 
            X_test_num = X_test[numerical_columns]
            X_test_cat = X_test[categorical_columns]

            logging.info("Split the numerical and categorical Datasets (X_num, X_cat) for train and test completed")



            # Imputing Data (Null Values)
            X_train_num_imputed, X_test_num_imputed = imputing_numerical(X_train_num, X_test_num)
            X_train_cat_imputed, X_test_cat_imputed = imputing_categorical(X_train_cat), imputing_categorical(X_test_cat)
            logging.info("Imputing both numerical and categorical data completed")
            
            logging.info("Getting into Feature Engineering")
            # Feature engineering 
            # Encoding Categorical Features
            onehot_cols = ['Ascites', 'Edema']  # Replace with your one-hot encoded column names
            ordinal_cols = ['Drug', 'Sex', 'Hepatomegaly','Spiders' ]  # Replace with your ordinal encoded column names
            X_train_cat_encoded, X_test_cat_encoded = encoding_mixed(X_train_cat_imputed, X_test_cat_imputed, onehot_cols, ordinal_cols)
            logging.info("Categorical Features encoding completed")
            # Target Encoding 
            Y_train_final = target_encoding(Y_train)
            Y_test_final = target_encoding(Y_test)
            logging.info("Target encoding completed")

            # Scaling numerical Features
            X_train_tf, X_test_tf = scaling(X_train_num_imputed, X_test_num_imputed) 
            logging.info("Scaling numerical features completed")
            # Final concatenation of numerical and categorical features into one Dataset
            X_train_final = pd.concat([X_train_tf.reset_index().drop('index', axis=1), X_train_cat_encoded.reset_index().drop('index', axis=1)], axis=1)
            X_test_final = pd.concat([X_test_tf.reset_index().drop('index', axis=1), X_test_cat_encoded.reset_index().drop('index', axis=1)], axis=1)
            logging.info("Final concatenation completed")


            return (
                X_train_final,
                X_test_final,
                Y_train_final,
                Y_test_final
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    datatransfo = DataTransformation()
    X_train_final, X_test_final, Y_train_final, Y_test_final = datatransfo.initiate_data_transformation(train_path = 'artifact/mcc_train.csv',  test_path = 'artifact/mcc_test.csv')
    print(X_train_final.shape)

       

