import sys
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.positive: int = 1
        self.negative: int = 0
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class TextModel:
    def __init__(self, vectorizer: TfidfVectorizer, trained_model_object: object):
        """
        :param vectorizer: Pre-trained TfidfVectorizer object
        :param trained_model_object: Trained ML model object
        """
        self.vectorizer = vectorizer
        self.trained_model_object = trained_model_object

    def predict(self, texts: list) -> DataFrame:
        """
        Accepts raw text input, transforms it using the TF-IDF vectorizer, and predicts the output.
        :param texts: List of raw text inputs
        """
        try:
            logging.info("Starting text transformation and prediction process.")
            
            # Transform text input using the trained vectorizer
            transformed_text = self.vectorizer.transform(texts)
            
            # Perform prediction using the trained model
            logging.info("Generating predictions using the trained model.")
            predictions = self.trained_model_object.predict(transformed_text)
            
            return predictions
        
        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
