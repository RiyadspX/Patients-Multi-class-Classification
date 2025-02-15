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

#### 1. Imputing Nan Values #####

def imputing_numerical(X_train_num, X_test_num, n_neighbors=10):
        imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state = 0)
        #imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42)
        #imputer = KNNImputer(n_neighbors=n_neighbors)
        
        imputer.fit(X_train_num)
        X_train_num_imputed = imputer.transform(X_train_num)
        X_test_num_imputed = imputer.transform(X_test_num)

        X_train_num_imputed = pd.DataFrame(X_train_num_imputed)
        X_train_num_imputed.columns = X_train_num.columns
        X_test_num_imputed = pd.DataFrame(X_test_num_imputed)
        X_test_num_imputed.columns = X_test_num.columns
        return X_train_num_imputed, X_test_num_imputed

# Imputing with the mode 
def imputing_categorical(df_fill):
    for col in df_fill.columns:
        if df_fill[col].isna().any():
            df_fill.loc[:, col] = df_fill[col].fillna(df_fill[col].mode()[0])
    return df_fill

# Imputing with a new category = Missing
def imputing_categorical_with_missing(df_fill):
    for col in df_fill.columns:
        if df_fill[col].isna().any():
            df_fill[col] = df_fill[col].fillna('Missing')
    return df_fill

#### 2. Encoding #####

def encoding_mixed(X_train_cat_imputed, X_test_cat_imputed):
    # Initialize lists to store column names for one-hot and ordinal encoding
    onehot_cols = []
    ordinal_cols = []
    
    # Loop through the columns to classify them
    for col in X_train_cat_imputed.columns:
        unique_vals = X_train_cat_imputed[col].nunique()
        
        if col.lower() == 'stage':  
            continue  # Do not encode 'Stage'
        elif unique_vals == 2:  
            ordinal_cols.append(col)  # Binary categorical → Ordinal encoding
        else:
            onehot_cols.append(col)  # Other categorical → One-hot encoding
    
    # Define the transformers for one-hot and ordinal encoding
    transformers = []
    
    if onehot_cols:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols))
    
    if ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols))
    
    # Create the ColumnTransformer
    column_transformer = ColumnTransformer(transformers, remainder='passthrough')

    # Fit the transformer on the training set and transform both datasets
    X_train_encoded = column_transformer.fit_transform(X_train_cat_imputed)
    X_test_encoded = column_transformer.transform(X_test_cat_imputed)

    # Convert the transformed data back into a DataFrame with proper column names
    all_columns = []
    if onehot_cols:
        onehot_feature_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(onehot_cols)
        all_columns.extend(onehot_feature_names)
    
    all_columns.extend(ordinal_cols)
    all_columns.extend([col for col in X_train_cat_imputed.columns if col not in onehot_cols + ordinal_cols])

    # Convert NumPy arrays to DataFrame with numerical values
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=all_columns, index=X_train_cat_imputed.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=all_columns, index=X_test_cat_imputed.index)

    # Ensure all non-'Stage' columns are numeric
    for col in X_train_encoded.columns:
        if col.lower() != 'stage':
            X_train_encoded[col] = pd.to_numeric(X_train_encoded[col], errors='coerce')
            X_test_encoded[col] = pd.to_numeric(X_test_encoded[col], errors='coerce')

    return X_train_encoded, X_test_encoded

def target_encoding(Y):
    code = {'C':0,'CL':2, 'D':1}
    Y = Y.map(code)
    return Y

#### .3 Scaling #####

def scaling(X_train, X_test):
    X_train_tf = X_train.copy()
    X_test_tf = X_test.copy()

    log_features = ['N_Days', 'Bilirubin', 'Cholesterol', 'Alk_Phos', 'SGOT', 'Tryglicerides']
    numerical_features = [col for col in X_train.select_dtypes('float').columns if col not in log_features]
    
    # only scale numeric varaibles in this case rather than the dummy variables for categories 
    rob = RobustScaler()
    X_train_tf.loc[:, numerical_features] = rob.fit_transform(X_train_tf.loc[:, numerical_features])
    X_test_tf.loc[:, numerical_features] = rob.transform(X_test_tf.loc[:, numerical_features])

    # log 
    X_train_tf.loc[:, log_features] = X_train_tf.loc[:, log_features].apply(np.log1p)
    X_test_tf.loc[:, log_features] = X_test_tf.loc[:, log_features].apply(np.log1p)


    return X_train_tf, X_test_tf

#### 4. feature Engineering #####

def feature_engineering_numerical(X_num):
    
    #X_num.loc[:, 'N_Years'] = X_num['N_Days'] / 365
    X_num['Bilirubin / Albumin'] = X_num['Bilirubin'] / X_num['Albumin']
    X_num['SGOT / Platelets'] = X_num['SGOT'] / X_num['Platelets']
    X_num['ISH'] = 0.4 * X_num['Cholesterol'] + 0.4 * X_num['Tryglicerides'] + 0.2 * X_num['Albumin']
    X_num['Coagulation_Score'] = X_num['Prothrombin'] / X_num['Platelets']

    return X_num

def feature_engineering_categorical(X_cat):
    #X_cat["Severe_Water_Retention"] = (X_cat["Edema"]  & X_cat["Ascites"])
    return X_cat


#### 5. Basic Preprocessing #####


def preprocess_train(data): 
    # set the column Stage to object 
    data['Stage'] = data['Stage'].astype(object)

    # replacing weird values of variables 
    data.loc[data['Hepatomegaly'] == '119.35', 'Hepatomegaly'] = data['Hepatomegaly'].mode()[0]
    data.loc[data['Ascites'] == 'S', 'Ascites'] = data['Ascites'].mode()[0]
    data.loc[data['Ascites'] == 'D-penicillamine', 'Ascites'] = data['Ascites'].mode()[0]

    # Delete where N_Days > Age
    data = data[data['Age']>data['N_Days']] 

    return data


def preprocess_test(data): 
    # set the column Stage to object 
    data['Stage'] = data['Stage'].astype(object)

    # replacing weird values of variables 
    data.loc[data['Hepatomegaly'] == '119.35', 'Hepatomegaly'] = data['Hepatomegaly'].mode()[0]
    data.loc[data['Ascites'] == 'S', 'Ascites'] = data['Ascites'].mode()[0]
    data.loc[data['Ascites'] == 'D-penicillamine', 'Ascites'] = data['Ascites'].mode()[0]

    data.loc[data['Spiders'] == 'C', 'Spiders'] = data['Spiders'].mode()[0]
    data.loc[data['Drug'] == 'Drug', 'Drug'] = data['Drug'].mode()[0]
    # Delete where N_Days > Age
    #data = data[data['Age']>data['N_Days']] 

    return data



# 1. Create a data transfo config to configure path 
#@dataclass
#class DataTransformationConfig:
    #preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")
    
# Creating Data transfo class 

class DataTransformation:
# Initialise the path config within the init
    def __init__(self):
        #self.data_transformation_config = DataTransformationConfig()
        pass

    def initiate_data_transformation_train(self, data_path):
        try : 
            dataset = pd.read_csv(data_path)
            logging.info("loading data completed")

            dataset = preprocess_train(dataset)
            logging.info("Read Data completed")

            # Specifiyng columns
            target_column_name = "Status"
            numerical_columns = ['N_Days', 'Bilirubin', 'Cholesterol', 
                                'Albumin', 'Copper', 'Alk_Phos',
                                'SGOT', 'Tryglicerides', 'Platelets', 
                                'Prothrombin', 'Age']
            categorical_columns = ['Stage','Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
            columns_delete = ['id']


            # Splitting the matrices 
            Y = dataset[target_column_name]
            X = dataset.drop([target_column_name] + columns_delete, axis=1)
            logging.info("Splitting into features and target completed")


            # Preparing the X_train, Y_train, X_test, Y_test with stratified method
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in sss.split(X, Y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            
            logging.info("Splitting into features and target completed")
            #X_train, X_test, Y_train, Y_test = train_test_split(
            #X, Y, test_size=0.2, random_state=8, stratify=Y)


            # SPLITTING Categorical and numerical 
            # Train 
            X_train_num = X_train[numerical_columns]
            X_train_cat = X_train[categorical_columns]
            # Test 
            X_test_num = X_test[numerical_columns]
            X_test_cat = X_test[categorical_columns]

            logging.info("Splitting Features into numerical and categorical completed")


            # Imputing Data (Null Values)
            # For numerical Features 
            X_train_num_imputed, X_test_num_imputed = imputing_numerical(X_train_num, X_test_num)

            # For Categorical Features
            X_train_cat_imputed, X_test_cat_imputed = imputing_categorical_with_missing(X_train_cat), imputing_categorical_with_missing(X_test_cat)
            #X_train_cat_imputed, X_test_cat_imputed = imputing_categorical(X_train_cat), imputing_categorical(X_test_cat)
            logging.info("Imputation completed")
            
            

            # Feature engineering 

            # Encoding Categorical Features
            onehot_cols = ['Edema']  # one-hot encoded column names
            ordinal_cols = ['Drug', 'Sex', 'Stage', 'Hepatomegaly','Spiders','Ascites']  # ordinal encoded column names
            X_train_cat_encoded, X_test_cat_encoded = encoding_mixed(X_train_cat_imputed, X_test_cat_imputed)#, onehot_cols, ordinal_cols)
            logging.info("Features encoding completed")

            # Target Encoding 
            Y_train_final = target_encoding(Y_train)
            Y_test_final = target_encoding(Y_test)
            logging.info("Target encoding completed")

            # Custom Feature functions 
            X_train_num_ft, X_test_num_ft = feature_engineering_numerical(X_train_num_imputed), feature_engineering_numerical(X_test_num_imputed) 
            X_train_cat_ft, X_test_cat_ft = feature_engineering_categorical(X_train_cat_encoded), feature_engineering_categorical(X_test_cat_encoded)
            logging.info("Feature engineering completed")


            # Scaling numerical Features
            X_train_tf, X_test_tf = scaling(X_train_num_ft, X_test_num_ft)
            logging.info("Scaling completed")

            # Final concatenation of numerical and categorical features into one Dataset

            X_train_final = pd.concat([X_train_tf.reset_index().drop('index', axis=1), X_train_cat_ft.reset_index().drop('index', axis=1)], axis=1)
            X_test_final = pd.concat([X_test_tf.reset_index().drop('index', axis=1), X_test_cat_ft.reset_index().drop('index', axis=1)], axis=1)
            logging.info("Final concatenation completed")
            # droping some columns 
            #X_train_final = X_train_final.drop(['Drug'], axis=1)
            #X_test_final = X_test_final.drop(['Drug'], axis=1)

            logging.info("Data transformation completed for both train and test set")
            return (
                X_train_final,
                X_test_final,
                Y_train_final,
                Y_test_final
            )
        except Exception as e:
            raise CustomException(e, sys)

    def transform_test_data(self, data_path):

        try:
            test = pd.read_csv(data_path) 

            test = preprocess_test(test)
            #test_dataset.loc[test_dataset['Spiders'] == 'C', 'Spiders'] = test_dataset['Spiders'].mode()[0]
            #test_dataset.loc[test_dataset['Drug'] == 'Drug', 'Drug'] = test_dataset['Drug'].mode()[0]


            # Specifying columns for the transformation pipeline
            numerical_columns = ['N_Days', 'Bilirubin', 'Cholesterol', 
                                'Albumin', 'Copper', 'Alk_Phos',
                                'SGOT', 'Tryglicerides', 'Platelets', 
                                'Prothrombin', 'Age']
            categorical_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema','Stage']
            columns_delete = ['id']  # Exclude the 'id' column in the test set

            # Remove 'id' column from test dataset
            X_test = test.drop(columns_delete, axis=1)
            

            # Split into numerical and categorical columns
            X_test_num = X_test[numerical_columns]
            X_test_cat = X_test[categorical_columns]

            # Imputation
            X_test_num_imputed = imputing_numerical(X_test_num, X_test_num)[1]  # Impute based on the test data itself
            X_test_cat_imputed = imputing_categorical_with_missing(X_test_cat)  # Impute categorical features

            # Encoding categorical features
            onehot_cols = ['Edema','Hepatomegaly','Spiders','Ascites']  # one-hot encoded column names
            ordinal_cols = ['Drug', 'Sex', 'Stage'] 
            X_test_cat_encoded = encoding_mixed(X_test_cat_imputed, X_test_cat_imputed)[1]#, onehot_cols, ordinal_cols)[1]  # Only need the transformed test data
            
            
            # Feature engineering for numerical and categorical features
            X_test_num_ft = feature_engineering_numerical(X_test_num_imputed)
            X_test_cat_ft = feature_engineering_categorical(X_test_cat_encoded)
            
            

            # Scaling numerical features
            X_test_tf = scaling(X_test_num_ft, X_test_num_ft)[1]  # Only scale based on test data (no fitting here)

            # Final concatenation of transformed numerical and categorical features
            X_test_final = pd.concat([X_test_tf.reset_index(drop=True), 
                                    X_test_cat_ft.reset_index(drop=True)], axis=1)
            
            X_test_final['Stage'] = X_test_final['Stage'].astype(float) 
            #X_test_final = X_test_final.drop(['Drug'], axis=1)

            return X_test_final
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    datatransfo = DataTransformation()
    X_train_final, X_test_final, Y_train_final, Y_test_final = datatransfo.initiate_data_transformation_train(data_path = 'artifact/mcc_data.csv')
    print(X_train_final.shape)

    X_sub = datatransfo.transform_test_data(data_path='Datasets/test.csv')
    print(X_sub.shape)
    print(X_sub.columns)

       

