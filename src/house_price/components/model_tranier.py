import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.house_price.exception import CustomException
from src.house_price.logger import logging
from src.house_price.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Log transform target 
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(objective="reg:squarederror"),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "LinearRegression": {},
                "DecisionTree": {
                    "max_depth": [None, 5, 10, 20]
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "GradientBoosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [100, 200]
                },
                "XGBoost": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [100, 200]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.05]
                }
            }

            results = {}
            best_model = None
            best_model_name = ""
            best_r2 = -np.inf
            best_metrics = None

            print("\n----- MODEL EVALUATION --------\n")

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                if params[model_name]:
                    grid = GridSearchCV(
                        model,
                        params[model_name],
                        cv=5,
                        scoring="r2",
                        n_jobs=-1
                    )
                    grid.fit(X_train, y_train)
                    trained_model = grid.best_estimator_
                else:
                    trained_model = model
                    trained_model.fit(X_train, y_train)

                y_pred = trained_model.predict(X_test)

                rmse, mae, r2 = self.evaluate_metrics(y_test, y_pred)

                results[model_name] = {
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                }

                print(f"{model_name}")
                print(f"   RMSE : {rmse:.4f}")
                print(f"   MAE  : {mae:.4f}")
                print(f"   R2   : {r2:.4f}\n")

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = trained_model
                    best_model_name = model_name
                    best_metrics = (rmse, mae, r2)

            if best_model is None:
                raise CustomException("No suitable model found", sys)

            print("------ BEST MODEL -----")
            print(f"Model Name : {best_model_name}")
            print(f"Best RMSE  : {best_metrics[0]:.4f}")
            print(f"Best MAE   : {best_metrics[1]:.4f}")
            print(f"Best R2    : {best_metrics[2]:.4f}")
            print("--------------------------------\n")

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best R2 Score: {best_r2:.4f}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_r2

        except Exception as e:
            raise CustomException(e, sys)
