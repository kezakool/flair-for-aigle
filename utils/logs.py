
import re
from datetime import datetime
import logging
import os
from typing import Union
import json
from utils.s3 import load_s3_json_file



def clear_logger_handlers(logger_name):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()  # Remove existing handlers
    logger.propagate = False  # Prevent propagation to the root logger
    
def configure_logging(log_file_path, progression_file_path, level=logging.INFO):
    """Configure logging for the application, including AWS boto3, botocore, and mmengine."""
    
    global S3_PROGRESSION_FILE_PATH, TMP_PROGRESSION_FILE_PATH

    S3_PROGRESSION_FILE_PATH = progression_file_path
    tmp_local_progression_file = os.path.join(log_file_path.rsplit('/',1)[0],'tmp_progression.json')
    TMP_PROGRESSION_FILE_PATH = tmp_local_progression_file
    
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    # Create stream handler (for terminal output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log startup message
    logging.info("Logging system configured.")

    # Configure specific loggers
    for module in ["gdal2tiles", "gdalbuildvrt", "gdalwarp", "boto3", "botocore"]:
        clear_logger_handlers(module)
        module_logger = logging.getLogger(module)
        module_logger.setLevel(level)
        module_logger.propagate = True
        logging.info(f"{module} logger integrated.")

    mmengine_logger = logging.getLogger("pytorch_lightning")
    mmengine_logger.setLevel(logging.INFO)
    mmengine_logger.propagate = True
    logging.info(f"pytorch_lightning logger integrated.")
    
    if progression_file_path:
        logging.info(f"Airflow run progression config detected! Infos stored in s3 : {progression_file_path}")
        progression_data = {
            "timestamp": str(datetime.now()),
            "status": "initializing",
            "progress": 0
        }
        # Ensure file is initialized
        with open(tmp_local_progression_file, "w", encoding="utf-8") as pf:
            pf.write(json.dumps(progression_data) + "\n")
        logging.info(f"Progression file initialized: {tmp_local_progression_file}")
        load_s3_json_file(tmp_local_progression_file, progression_file_path)

        
def update_progress(progress: Union[int, float], status: str):
    """Append a status line in the progression file."""
    global TMP_PROGRESSION_FILE_PATH, S3_PROGRESSION_FILE_PATH
    if S3_PROGRESSION_FILE_PATH:
        progression_data = {
            "timestamp": str(datetime.now()),
            "status": status,
            "progress": progress
        }
        with open(TMP_PROGRESSION_FILE_PATH, "a", encoding="utf-8") as pf:
            pf.write(json.dumps(progression_data) + "\n")
        load_s3_json_file(TMP_PROGRESSION_FILE_PATH, S3_PROGRESSION_FILE_PATH)
    else:
        logging.debug("Progression file path not set. Skipping progress update.")