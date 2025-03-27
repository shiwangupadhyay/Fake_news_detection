import sys
import numpy as np
import pandas as pd
import pickle

from src.constants import MAX_FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Read CSV file and return as DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
    
    @staticmethod
    def tfidf_vectorize(corpus, max_features=MAX_FEATURES):
        """Convert corpus into TF-IDF vectors."""
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(corpus)
        return X, vectorizer
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
            
            # Ensure processed text column exists
            if 'stemmed_tokens' not in train_df.columns or 'stemmed_tokens' not in test_df.columns:
                raise Exception("Processed text column missing in dataset.")
            
            # Convert text column to corpus
            train_corpus = train_df['stemmed_tokens'].astype(str).tolist()
            test_corpus = test_df['stemmed_tokens'].astype(str).tolist()
            
            # Apply TF-IDF transformation
            logging.info("Applying TF-IDF Vectorization")
            X_train, vectorizer = self.tfidf_vectorize(train_corpus)
            X_test = vectorizer.transform(test_corpus)
            logging.info(f"TF-IDF Transformation Done: Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
            
            # Save transformation objects
            save_object(self.data_transformation_config.transformed_object_file_path, vectorizer)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=X_train.toarray())
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=X_test.toarray())
            logging.info("Saved transformed data and vectorizer")
            
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys)
