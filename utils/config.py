import argparse
import json
import os
import boto3
import botocore
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads configuration from a JSON file."""
    if "s3://" in config_path:
        logger.info(f"s3 config detected - downloading file from {config_path}")
        run_folder = os.getenv('RUN_FOLDER')
        run_log_folder = os.path.join(run_folder, 'logs')
        if not os.path.isdir(run_log_folder):
            os.mkdir(run_log_folder)
        local_configs_path = os.path.join(run_log_folder,'configs')
        if not os.path.isdir(local_configs_path):
            os.mkdir(local_configs_path)   
        s3 = boto3.resource('s3')

        def extract_from_s3_path(config_path):
            config_path = config_path.replace('s3://','')
            bucket_name, key_path = config_path.split('/',1)
            _, filename = config_path.rsplit('/',1)
            return bucket_name, key_path, filename
        
        try:
            bucket_name, key_path, filename = extract_from_s3_path(config_path)
            config_path =  os.path.join(local_configs_path,filename)
            s3.Bucket(bucket_name).download_file(key_path,config_path)
            logger.info(f"s3 config detected - file downloaded to : {config_path}")
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info("The object does not exist.")
            else:
                raise Exception(f"Unable to download file from s3 bucket")
            
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def combine_args_with_priority(args, config):
    # This function combines args from argparse and config from JSON with args taking priority
    combined = vars(args).copy()  # Start with command-line args
    #combined.update({k: v for k, v in config.items() if getattr(args, k, None) is None})  # Use config values if not provided by args
    combined.update({k: v for k, v in config.items()})  # Use config values if provided else default arg
    return argparse.Namespace(**combined)

class ArgsObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)