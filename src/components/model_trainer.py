import sys
from typing import Tuple
import pandas as pd

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact, DataIngestionArtifact
from src.entity.estimator import TextModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Read CSV file and return as DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_model_object_and_report(self, X_train, y_train, X_test, y_test) -> Tuple[object, object]:
        try:
            logging.info("Training DecisionTreeClassifier with specified parameters")

            # Initialize DecisionTreeClassifier
            model = DecisionTreeClassifier(
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                max_depth=self.model_trainer_config._max_depth,
                criterion=self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state
            )

            # Fit the model
            logging.info("Model training going on...")
            model.fit(X_train, y_train)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Starting Model Trainer Component")
        try:
            train_feature = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_feature = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            train_label = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)['label']
            test_label = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)['label']
            logging.info("Loaded transformed train and test data.")

            trained_model, metric_artifact = self.get_model_object_and_report(X_train = train_feature, y_train = train_label, X_test=test_feature, y_test = test_label)
            logging.info("Model training and evaluation completed.")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded.")

            if accuracy_score(test_label, trained_model.predict(test_feature)) < self.model_trainer_config.expected_accuracy:
                logging.info("Trained model does not meet the expected accuracy threshold.")
                raise Exception("Trained model does not meet the expected accuracy threshold.")

            logging.info("Saving the trained model.")
            my_model = TextModel(vectorizer=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Model saved successfully.")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e