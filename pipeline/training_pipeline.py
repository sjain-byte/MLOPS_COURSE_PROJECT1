from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *

if __name__ == "__main__":
    ####1. DATA INGESTION

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ####2. DATA PROCESSING

    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()

    ####3. MODEL TRAINING
    trainer = ModelTraining(
            PROCESSED_TRAIN_DATA_PATH, 
            PROCESSED_TEST_DATA_PATH, 
            MODEL_OUTPUT_PATH, 
            "http://localhost:5000"  # Set your MLflow tracking URI
        )
    trainer.run()