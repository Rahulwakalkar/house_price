import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.house_price.utils import save_object
from src.house_price.exception import CustomException
from src.house_price.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X_df):
        """
        Builds preprocessing pipeline for House Price dataset
        """
        try:
            numerical_columns = X_df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_columns = X_df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded successfully")

            target_column_name = "SalePrice"

            # Drop ID column if exists
            if "Id" in train_df.columns:
                train_df = train_df.drop(columns=["Id"])
                test_df = test_df.drop(columns=["Id"])

            # Separate features and target FIRST
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            # Build transformer using ONLY features
            preprocessing_obj = self.get_data_transformer_object(X_train)

            logging.info("Applying preprocessing")

            
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

         # Convert sparse matrix to dense if needed
            if hasattr(X_train_arr, "toarray"):
              X_train_arr = X_train_arr.toarray()
            if hasattr(X_test_arr, "toarray"):
              X_test_arr = X_test_arr.toarray()

           # Ensure features are 2D
            if X_train_arr.ndim == 1:
              X_train_arr = X_train_arr.reshape(-1, 1)
            if X_test_arr.ndim == 1:
                X_test_arr = X_test_arr.reshape(-1, 1)


            # Ensure target is 2D
            y_train_arr = y_train.to_numpy().reshape(-1, 1)
            y_test_arr = y_test.to_numpy().reshape(-1, 1)

            # Combine features and target
            train_arr = np.hstack((X_train_arr, y_train_arr))
            test_arr = np.hstack((X_test_arr, y_test_arr))

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
