import os
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path: {file_path}")
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Yaml file read successfully")
            return config
    except Exception as e:
        logger.error("Error reading the yaml file")
        raise CustomException("Error while reading YAML file", e)

def load_data(file_path):
    try:
        logger.info("Loading the data file")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading the data file")
        raise CustomException("Error while loading data", e)