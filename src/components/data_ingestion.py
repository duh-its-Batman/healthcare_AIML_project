import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
	train_data_path: str = os.path.join("artifacts", "train.csv")
	test_data_path: str = os.path.join("artifacts", "test.csv")
	raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
	def __init__(self):
		self.ingestion_config = DataIngestionConfig()

	def initiate_data_ingestion(self):
		logging.info("Data ingestion method or component has been initiated.")
		try:
			files_list = ["Normal.csv", "Type_H.csv", "Type_S.csv"]  # list of the .csv files we want to ingest
			df_list = []
			main_df = pd.DataFrame()
			if len(df_list) == 0:
				for file in files_list:
					df = pd.read_csv("D:/art_intel/portfolio_projects/healthcare_AIML_project/data_and_notebook/data/{}".format(file)) 
					df_list.append(df)
				
				main_df = pd.concat(df_list)    # merging all the dataframes into one
				logging.info("Dataset ingested and prepared for use as a DataFrame.")

			else:
				logging.info("Dataset already prepared.")

			os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
			main_df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
			logging.info("Train-test split initiated.")
			train_set, test_set = train_test_split(main_df, test_size = 0.25, random_state = 24)
			train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
			test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
			logging.info("Data ingestion completed.")

			return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
			
		except Exception as e:
			raise CustomException(e, sys)


if __name__ == "__main__":
	obj = DataIngestion()
	obj.initiate_data_ingestion()