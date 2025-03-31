import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        self.selected_features = None  # Store selected features

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"Created directory: {self.processed_dir}")

    def preprocess_data(self, df):
        try:
            logger.info("Preprocessing the data")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], errors='ignore', inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Label Encoding categorical columns")
            label_encoder = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                if col in df.columns:
                    df[col] = label_encoder.fit_transform(df[col])
                    mappings[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

            logger.info(f"Label Encoding mappings: {mappings}")

            logger.info("Handling skewness")
            skew_threshold = self.config["data_processing"]["skew_threshold"]
            skewness = df[num_cols].apply(lambda x: x.skew() if x.dtype != 'O' else 0)

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise CustomException(f"Error in preprocessing data: {e}")

    def balance_data(self, df):
        try:
            logger.info("Balancing the data")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            df_balanced["booking_status"] = y_resampled
            logger.info("Data balanced successfully")
            return df_balanced
        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException(f"Error in balancing data: {e}")

    def select_features(self, df):
        try:
            logger.info("Selecting features")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": feature_importance})

            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
            num_features_to_select = self.config["data_processing"]["no_of_features"]
            self.selected_features = top_features_importance_df["feature"].head(num_features_to_select).tolist()

            selected_df = df[self.selected_features + ["booking_status"]]
            logger.info(f"Top {num_features_to_select} features selected: {self.selected_features}")
            return selected_df
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException(f"Error in feature selection: {e}")

    def apply_feature_selection_to_test(self, df):
        try:
            if self.selected_features is None:
                raise CustomException("Feature selection must be run before applying to test data.")

            logger.info("Applying feature selection to test data")
            missing_features = [col for col in self.selected_features if col not in df.columns]

            # Add missing features with default value (0)
            for col in missing_features:
                df[col] = 0

            return df[self.selected_features]
        except Exception as e:
            logger.error(f"Error in applying feature selection to test data: {e}")
            raise CustomException(f"Error in applying feature selection to test data: {e}")

    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error in saving data: {e}")
            raise CustomException(f"Error in saving data: {e}")

    def process(self):
        try:
            logger.info("Starting data processing")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Loaded train and test data")
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            logger.info("Preprocessed train and test data")
            train_df = self.balance_data(train_df)

            logger.info("Balanced train data")
            train_df = self.select_features(train_df)

            logger.info("Applying feature selection to test data")
            test_df = self.apply_feature_selection_to_test(test_df)

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException(f"Error in data processing: {e}")

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
    logger.info("Data processing script executed successfully")
