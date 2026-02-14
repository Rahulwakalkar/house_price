import os
import sys
import pandas as pd
from dataclasses import dataclass
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.house_price.exception import CustomException
from src.house_price.logger import logging


# Load environment variables
from pathlib import Path

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _get_mysql_connection(self):
        try:
            host = os.getenv("MYSQL_HOST")
            user = os.getenv("MYSQL_USER")
            password = os.getenv("MYSQL_PASSWORD")
            database = os.getenv("MYSQL_DATABASE")
            port = os.getenv("MYSQL_PORT", "3306")

            if not all([host, user, password, database]):
                raise ValueError("Missing database credentials in .env file")

            connection_string = (
                f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            )

            engine = create_engine(connection_string)
            logging.info("MySQL connection created successfully")

            return engine

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")

            engine = self._get_mysql_connection()

            # Replace with your actual table name
            query = "SELECT * FROM train"
            df = pd.read_sql(query, engine)

            if df.empty:
                raise ValueError("Fetched dataset is empty")

            logging.info("Data fetched successfully from MySQL")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed Successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error in Data Ingestion")
            raise CustomException(e, sys)
