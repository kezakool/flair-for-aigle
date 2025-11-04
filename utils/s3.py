import os
import boto3
import shutil
from pathlib import Path
import glob

import zipfile
import py7zr
import subprocess
import pandas as pd
from utils.utils import generate_timestamp

from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)


def load_s3_json_file(local_file_path, s3_file_path):

    session = boto3.session.Session(profile_name='default')
    s3_client = session.client(service_name='s3',region_name='fr-par',use_ssl=True,endpoint_url='http://s3.fr-par.scw.cloud')
    bucket_name, file_key = s3_file_path.replace('s3://','').split('/',1)
    s3_client.upload_file(local_file_path, bucket_name, file_key)
    logging.info(f"file {local_file_path} uploaded to s3 : {s3_file_path}")

def prepare_local_model_folder(run_folder,model_id):
    """
    Prepares the local model folder by downloading the model from S3 if it is not already present.

    Parameters:
    run_folder (str): The path to the run folder where the model will be stored.
    model_id (int): The ID of the model to be prepared.

    Returns:
    tuple: A tuple containing the file paths to the model configuration, checkpoint, and threshold files.

    Raises:
    subprocess.CalledProcessError: If the S3 download command fails.

    Example:
    >>> prepare_local_model_folder('/path/to/run_folder', 123)
    ('/path/to/run_folder/models/123/model_config.py', '/path/to/run_folder/models/123/model_onnx_ckpt', '/path/to/run_folder/models/123/best_thresholds.yaml')
    """
    logger.info(f"Initializing ml model configuration from id : {model_id} ...")
    
    df_model = pd.read_sql(f"select * from detections.model where id = {model_id}",con=os.getenv('DB_STRING_PROD'))
    model_infos = df_model.iloc[0]
    model_s3_path = model_infos["model_path"]
    models_path = os.path.join(run_folder,"models")
    
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    
    model_path = os.path.join(models_path,str(model_id))
    if not os.path.isdir(model_path):
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        logger.info(f"Downloading model from {model_s3_path} to {model_path}....")
        process = subprocess.run(["aws","s3","cp",model_s3_path,model_path,"--recursive"] ,check=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True)
        # Log stdout
        if process.stdout:
            for line in process.stdout.splitlines():
                logger.info(f"S3 SYNC : {line}")  # Log each line
        # Log stderr (if any errors/warnings)
        if process.stderr:
            for line in process.stderr.splitlines():
                logger.warning(f"S3 SYNC ERROR : {line}")
    else:
            logger.info(f"Ml model config found locally at : {model_path}")  

    # -------------------------------
    # Find model checkpoint file
    # -------------------------------
    extensions = ["*.pt", "*.ckpt", "*.safetensors"]
    model_ckpt_path = None

    for ext in extensions:
        matches = glob.glob(os.path.join(model_path, "**", ext), recursive=True)
        if matches:
            model_ckpt_path = matches[0]
            logger.info(f"Model checkpoint found: {model_ckpt_path}")
            break

    if not model_ckpt_path:
        logger.error(f"No model checkpoint (.pt/.ckpt/.safetensors) found in {model_path}")
        raise FileNotFoundError(f"No checkpoint file found in {model_path}")

    # -------------------------------
    # Locate threshold configuration file
    # -------------------------------
    model_threshold_filepath = os.path.join(model_path, "best_thresholds.yaml")
    if not os.path.exists(model_threshold_filepath):
        logger.warning(f"Threshold config file not found at: {model_threshold_filepath}")

    return model_ckpt_path, model_threshold_filepath


def prepare_run_folder(experiment_run_folder, progression_file_path):
    """
    Prepares the run folder for an experiment by creating necessary subdirectories
    and configuring logging.

    Parameters:
    experiment_run_folder (str): The path to the experiment run folder.

    Returns:
    tuple: A tuple containing the paths to the run log folder and run result folder.
    """
    if not os.path.isdir(experiment_run_folder):
        os.makedirs(experiment_run_folder)
    
    run_log_folder = os.path.join(experiment_run_folder,'logs')
    run_result_folder = os.path.join(experiment_run_folder,'results')

    # Configure logging
    run_timestamp = generate_timestamp()
    
    if not os.path.isdir(run_log_folder):
        os.mkdir(run_log_folder)
    from utils.logs import configure_logging 
    configure_logging(os.path.join(run_log_folder,f"{run_timestamp}_segmentation-log.csv"), progression_file_path,level=logging.INFO)

    logger.info("Loading run configuration...")
    
    logger.info(f"Run Logs will be stored at : {run_log_folder}")    
    if not os.path.isdir(run_result_folder):
        os.mkdir(run_result_folder)
    logger.info(f"Run results will be stored at : {run_result_folder}")
    return run_log_folder, run_result_folder
    
def prepare_local_data_folder(bucket_name, aerial_archive_source_folder, db_topo_archive_source_file, experiment_data_folder, add_building_db_topo, use_remove_db_topo):
    """
    Prepares the local data folder by downloading and extracting necessary data from S3.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - aerial_archive_source_folder (str): The source folder in the S3 bucket for aerial images.
    - db_topo_archive_source_file (str): The source file in the S3 bucket for DB topo data.
    - experiment_data_folder (str): The local folder where the experiment data will be stored.
    - add_building_db_topo (bool): Flag to indicate whether to add BD topo building data.
    - use_remove_db_topo (bool): Flag to indicate whether to use remove BD topo data (roads).
    Returns:
    - tuple: A tuple containing the path to the source folder and the path to the BD topo files.
    """
    
    session = boto3.session.Session(profile_name='default')
    s3_client = session.client(service_name='s3',region_name='fr-par',use_ssl=True,endpoint_url='http://s3.fr-par.scw.cloud')
    
    logger.info(f"Run datas will be stored at : {experiment_data_folder}")
    
    dl_work_folder = os.path.join(experiment_data_folder, 'work')
    source_folder = os.path.join(experiment_data_folder, 'sources-raw')
    db_cache_folder = os.path.join(experiment_data_folder, 'db-cache')
    # Set up workdir preppath
    if not os.path.isdir(dl_work_folder):
        os.makedirs(dl_work_folder)
    if not os.path.isdir(db_cache_folder):
        os.makedirs(db_cache_folder)
    
    def check_source_status(source_folder):
        # if one or more of images folders miss files, download archives from S3 and extract to sources
        download_and_extract_sources = False

        if not os.path.isdir(source_folder):
            download_and_extract_sources = True
            os.mkdir(source_folder)
        elif len(os.listdir(source_folder))==0:
            download_and_extract_sources = True
        else :
            download_and_extract_sources = False
        return download_and_extract_sources
    
    download_and_extract_sources = check_source_status(source_folder)
    
    if download_and_extract_sources:
        download_extract_aerials(s3_client, bucket_name, aerial_archive_source_folder, dl_work_folder, dl_work_folder)
    
    move_files_to_dest_folder(dl_work_folder, source_folder, extension='.jp2')
    
    if add_building_db_topo:
        download_and_extract_sources = check_source_status(db_cache_folder)
        if download_and_extract_sources:
            download_extract_db_topo(s3_client,bucket_name, db_topo_archive_source_file,dl_work_folder, dl_work_folder) 
        move_files_to_dest_folder(dl_work_folder,db_cache_folder)
        
        building_db_topo_path = os.path.join(db_cache_folder,'BATIMENT.shp')
        if not os.path.isfile(building_db_topo_path):
            logging.warning("bd topo format old")
            building_db_topo_path = os.path.join(db_cache_folder,'BATI_INDIFFERENCIE.SHP')
            if not os.path.isfile(building_db_topo_path):
                logging.warning("bd topo not found")
    else:
        building_db_topo_path = ''
    
    use_remove_roads_db_topo = True
    if use_remove_roads_db_topo:
        download_and_extract_sources = check_source_status(db_cache_folder)
        if download_and_extract_sources:
            download_extract_db_topo(s3_client,bucket_name, db_topo_archive_source_file,dl_work_folder, dl_work_folder) 
        move_files_to_dest_folder(dl_work_folder,db_cache_folder)
        
        roads_db_topo_path = os.path.join(db_cache_folder,'TRONCON_DE_ROUTE.shp')
        if not os.path.isfile(roads_db_topo_path):
            logging.warning("bd topo format old")
            roads_db_topo_path = os.path.join(db_cache_folder,'ROUTE.SHP')
            if not os.path.isfile(roads_db_topo_path):
                logging.warning("bd topo not found")
    else:
        roads_db_topo_path = ''
    
    use_remove_waters_db_topo = True
    if use_remove_waters_db_topo:
        download_and_extract_sources = check_source_status(source_folder)
        if download_and_extract_sources:
            download_extract_db_topo(s3_client,bucket_name, db_topo_archive_source_file,dl_work_folder, dl_work_folder) 
        move_files_to_dest_folder(dl_work_folder,db_cache_folder)
        
        waters_db_topo_path = os.path.join(db_cache_folder,'SURFACE_HYDROGRAPHIQUE.shp')
        if not os.path.isfile(waters_db_topo_path):
            logging.warning("bd topo format old")
            waters_db_topo_path = os.path.join(db_cache_folder,'SURFACE_EAU.SHP')
            if not os.path.isfile(waters_db_topo_path):
                logging.warning("bd topo not found")
    else:
        waters_db_topo_path = ''
    
    return source_folder, building_db_topo_path, roads_db_topo_path, waters_db_topo_path

    
def upload_run_traces_to_s3(s3_runs_path,experiment_run_folder,image_set_name):
    # Sync results and logs to S3
    logger.info(f"starting run synchronization to s3 folder : {s3_runs_path}...")
    
    process = subprocess.run(['aws', 's3', 'sync', experiment_run_folder, s3_runs_path + '/' + image_set_name] ,check=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True)
    # Log stdout
    if process.stdout:
        for line in process.stdout.splitlines():
            logger.info(f"S3 SYNC : {line}")  # Log each line
    # Log stderr (if any errors/warnings)
    if process.stderr:
        for line in process.stderr.splitlines():
            logger.warning(f"S3 SYNC ERROR : {line}")
            
    logger.info(f"run data synchronized to s3 folder : {s3_runs_path + '/' + image_set_name} - Done")

def move_files_to_dest_folder(source_folder, target_folder, extension=''):
    """
    Move all .jp2 files from the source folder and its subfolders to the target folder.

    :param source_folder: The folder to search for .jp2 files.
    :param target_folder: The folder to move the .jp2 files to.
    """
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Walk through the source folder and its subfolders
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if extension== '' and not file.endswith('7z') :
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Construct the target file path
                target_file_path = os.path.join(target_folder, file)
                # Move the file
                shutil.move(file_path, target_file_path)
                logger.info(f"Moved: {file_path} -> {target_file_path}")
            else :
                if file.endswith(extension):
                    # Construct the full file path
                    file_path = os.path.join(root, file)
                    # Construct the target file path
                    target_file_path = os.path.join(target_folder, file)
                    # Move the file
                    shutil.move(file_path, target_file_path)
                    logger.info(f"Moved: {file_path} -> {target_file_path}")

def download_batch_archives(s3_client, bucket_name, archive_source_folder, local_temp_dir):
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=archive_source_folder)
    
    file_paths = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        file_name = key.split('/')[-1]
        local_path = os.path.join(local_temp_dir, file_name)
        
        if obj['Size']>0 :
            if not os.path.isfile(local_path):
                # Download the archive part
                logger.info(f"Downloading {file_name}...")
                s3_client.download_file(Bucket=bucket_name, Key=key, Filename=local_path)
            file_paths.append(local_path)
    return file_paths
        
def concatenate_and_extract(file_paths, extract_dir, local_temp_dir):
    # Assuming all parts are needed to be concatenated in order
    with open(os.path.join(local_temp_dir, 'archive.7z'), 'wb') as wfd:
        for f in sorted(file_paths):
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd, 1024*1024*10)
    logger.info("Extracting archive...")
    with py7zr.SevenZipFile(os.path.join(local_temp_dir, 'archive.7z'), mode='r') as archive:
        archive.extractall(path=extract_dir)
    logger.info(f"Extracted '{os.path.join(local_temp_dir, 'archive.7z')}' to '{extract_dir}'")
    #shutil.unpack_archive(os.path.join(local_temp_dir, 'archive.7z'), extract_dir)

def download_extract_aerials(s3_client, bucket_name, aerial_archive_source_folder, aerial_extract_dir, local_temp_dir):
    file_paths = download_batch_archives(s3_client, bucket_name, aerial_archive_source_folder, local_temp_dir)
    concatenate_and_extract(file_paths, aerial_extract_dir, local_temp_dir)

def download_extract_db_topo(s3_client, bucket_name, db_topo_archive_source_path,db_cache_path, local_temp_dir):
    local_path = os.path.join(local_temp_dir, db_topo_archive_source_path.rsplit('/',1)[1])
    if not os.path.isfile(local_path):
        # Download the archive part
        logger.info(f"Downloading {db_topo_archive_source_path}...")
        s3_client.download_file(Bucket=bucket_name, Key=db_topo_archive_source_path, Filename=local_path)

        with py7zr.SevenZipFile(local_path, mode='r') as archive:
            archive.extractall(path=db_cache_path)

def download_extract_pleiades(s3_client, bucket_name, pleiade_archive_source,pleiade_extract_dir, local_temp_dir):
    local_path = os.path.join(local_temp_dir, pleiade_archive_source.rsplit('/',1)[1])

    if not os.path.isfile(local_path):
        # Download the archive part
        logger.info(f"Downloading {pleiade_archive_source}...")
        s3_client.download_file(Bucket=bucket_name, Key=pleiade_archive_source, Filename=local_path)
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            # Extract all the contents into the directory
            zip_ref.extractall(pleiade_extract_dir)

def upload_extracted_jp2(s3_client, bucket_name, local_dir, dest_prefix):
    for filepath in glob.glob(local_dir + '**/*.jp2', recursive=True):
        s3_key = os.path.join(dest_prefix, os.path.relpath(filepath, start=local_dir))
        logger.info(f"Uploading {filepath} to {s3_key}...")
        s3_client.upload_file(Filename=filepath, Bucket=bucket_name, Key=s3_key)
        
def upload_directory_to_s3(local_directory, bucket_name, s3_prefix, s3_client, config):
    for root, dirs, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path)

            try:
                s3_client.upload_file(local_file_path, bucket_name, s3_key, Config=config)
                logger.info(f'Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}')
            except Exception as e:
                logger.info(f'Error uploading {local_file_path}: {e}')
