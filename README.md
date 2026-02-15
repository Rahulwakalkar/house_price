House Price Prediction-using machine learning


Machine Learning pipeline to predict house prices using the House Prices dataset.

The pipeline includes:

Data ingestion from MySQL
Data preprocessing & transformation
Model training & hyperparameter tuning
Model evaluation (RMSE, MAE, R²)
Best model selection

##project structure
house/
│
├── artifacts/                 # Saved models & preprocessors
│
├── src/
│   └── house_price/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       │
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
│
├── app.py
├── .env
├── requirements.txt
└── README.md



## Data Ingestion

Reads dataset from MySQL database
Splits into train & test sets
Saves raw data into artifacts/

## Data Transformation

Handles missing values
Median imputation (numerical)
Most frequent imputation (categorical)
OneHot Encoding for categorical features
Standard Scaling

Saves preprocessing object

## Model Training

Trains and evaluates multiple models:

Linear Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost
AdaBoost

## Evaluation Metrics

Each model is evaluated using:

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score

Best model is selected based on highest R² score.






## Required libraries:

numpy,pandas,scikit-learn, xgboost, pymysql, sqlalchemy, python-dotenv