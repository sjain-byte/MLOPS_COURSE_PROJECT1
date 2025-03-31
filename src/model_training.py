import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path, mlflow_tracking_uri):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

        # Set up MLflow tracking
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_name = "LightGBM_Classification"
        mlflow.set_experiment(self.experiment_name)

    def load_and_split_data(self):
        try:
            logger.info("Loading train and test data")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Ensure 'booking_status' is in train data but may not be in test
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            if 'booking_status' in test_df.columns:
                X_test = test_df.drop(columns=['booking_status'])
                y_test = test_df['booking_status']
            else:
                X_test, y_test = test_df, None  # Unlabeled test set

            logger.info("Data loaded and split successfully")
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException(e)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting LightGBM training")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                scoring=self.random_search_params["scoring"],
                cv=self.random_search_params["cv"], 
                verbose=self.random_search_params["verbose"],
                n_jobs=self.random_search_params["n_jobs"],
                random_state=self.random_search_params["random_state"],
            )
        
            logger.info("Starting Hyperparameter tuning")
            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters found: {best_params}")
            return best_lgbm_model

        except Exception as e:
            logger.error(f"Error in training LightGBM model: {e}")
            raise CustomException(e)
        
    def evaluate_model(self, model, X_test, y_test):
        if y_test is None:
            logger.warning("Test dataset has no labels. Skipping evaluation.")
            return {}

        try:
            logger.info("Evaluating the model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Evaluation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            logger.error(f"Error in evaluating model: {e}")
            raise CustomException(e)

    def save_model(self, model):
        try:
            logger.info("Saving the model")
            os.makedirs(self.model_output_path, exist_ok=True)
            model_path = os.path.join(self.model_output_path, "best_lgbm_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved at {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error in saving model: {e}")
            raise CustomException(e)

    def log_to_mlflow(self, model, metrics, model_path):
        try:
            logger.info("Logging experiment to MLflow")
            with mlflow.start_run():
                mlflow.log_params(self.params_dist)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                mlflow.log_artifact(model_path, artifact_path="model")
            logger.info("Successfully logged experiment to MLflow")
        except Exception as e:
            logger.error(f"Error in logging to MLflow: {e}")
            raise CustomException(e)

    def run(self):
        try:
            logger.info("Starting the model training pipeline")
            X_train, y_train, X_test, y_test = self.load_and_split_data()
            best_lgbm_model = self.train_lgbm(X_train, y_train)
            model_path = self.save_model(best_lgbm_model)
            
            # Evaluate only if labels exist
            metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)

            # Log results to MLflow
            self.log_to_mlflow(best_lgbm_model, metrics, model_path)

            logger.info("Model training pipeline completed successfully")
            return metrics
        
        except Exception as e:
            logger.error(f"Error in running the model training pipeline: {e}")
            raise CustomException(f"Failed during model training pipeline: {e}")
        
if __name__ == "__main__":
    try:
        trainer = ModelTraining(
            PROCESSED_TRAIN_DATA_PATH, 
            PROCESSED_TEST_DATA_PATH, 
            MODEL_OUTPUT_PATH, 
            "http://localhost:5000"  # Set your MLflow tracking URI
        )
        trainer.run()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise CustomException(e)
