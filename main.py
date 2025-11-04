import argparse
import sys
import json
from utils.config import *
from scripts.run_fast_aigle_segmentation import run_fast_aigle_segmentation

           
def main():

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Aigle - Construction Detection on PHR Images")
    parser.add_argument("-config", "--config", help="Path to the JSON configuration file", required=False)
    parser.add_argument("-process", "--process", choices=["prepare_training", "prepare_test", "evaluate", "predict", "run_best_detections","convert_dataset"], required=False)
    parser.add_argument("--run_id", help="airflow run id link to map between dags, is set in dag", required=False, default='local')
    parser.add_argument("--run_progression_filepath", help="airflow run progression file path to map between dags, is set in dag", required=False)
    
    # Fix for debug_mode flag
    parser.add_argument("-debug_mode", "--debug_mode", help="Activate debug mode", action='store_true')
    parser.add_argument("-images_type", "--images_type", choices=["pleiade", "aerial", "siatiles"], metavar="[IMAGE_TYPE]", required=False)
    
    parser.add_argument("-annotation_source_type", "--annotation_source_type", help="Annotations source, is from_custom_file or from_aigle_db", required=False)
    parser.add_argument("-model_type", "--model_type", help="Model type to load", required=False)
    parser.add_argument("-model_ckpt", "--model_ckpt", help="Best model loaded for inference on test set, if multiple, is separated with commas", required=False)
    parser.add_argument("-model_config", "--model_config", help="Model config from mmdetection training", required=False)
    parser.add_argument("-model_id", "--model_id", help="Model Id from Aigle detection schema", required=False)
    parser.add_argument("-images_folders", "--images_folders", help="Comma-separated list of directories with input images", required=False)
    parser.add_argument("-inference_folder", "--inference_folder", help="inference test set", default=None, required=False)
    parser.add_argument("-datasets_folder", "--datasets_folder", help="Output directory for Aigle experiments", required=False)
    parser.add_argument("-db_sources", "--db_sources", help="Directory with Aigle Parquet DB sources", required=False)
    
    parser.add_argument("-dataset_rootname", "--dataset_rootname", help="Root Name of the dataset folder", required=False)
    parser.add_argument("-testset_name", "--testset_name", help="Name of the test set folder (child of datasets folder)", required=False)
    parser.add_argument("-set_annotations_file", "--set_annotations_file", help="Annotations file for evaluation/inference in .parquet format", default=None, required=False)
    parser.add_argument("-set_images_file", "--set_images_file", help="File with geo CRS mapping information in .parquet format", default=None, required=False)
    parser.add_argument("-target_pixel_size_m", "--target_pixel_size_m", help="Pixel size for image resolution in dataset generation", default=None, required=False)
    parser.add_argument("-target_tile_size_px", "--target_tile_size_px", help="Target image size in pixels", default=256, required=False)
    parser.add_argument("-input_pixel_size_m", "--input_pixel_size_m", help="Input pixel size for image resolution in dataset generation", default=None, required=False)
    parser.add_argument("-input_tile_size_px", "--input_tile_size_px", help="Input image size in pixels", default=256, required=False)
    parser.add_argument("-tile_size_px", "--tile_size_px", help="Input image size in pixels", default=256, required=False)
    parser.add_argument("-input_crs", "--input_crs", help="Input CRS config", default='EPSG:2154', required=False)
    parser.add_argument("-target_crs", "--target_crs", help="Target CRS for generated images", default='EPSG:4326', required=False)
    parser.add_argument("-geozones_codes", "--geozones_codes", help="iso codes of department, epci, communes to filter input data (comma-separated); no filter if empty", default='all', required=False)
    parser.add_argument("-export_geozones_codes", "--export_geozones_codes", help="INSEE code export communes separates multiples with ','", default='all', required=False)
    parser.add_argument("-version", "--version", help="Dataset version attribute", required=False)
    parser.add_argument("-dataset_type", "--dataset_type", choices=["coco", "yolo"], help="Dataset export format (coco or yolo)", required=False)
    parser.add_argument("-classes_file", "--classes_file", help="Path to the classes file", required=False)
    parser.add_argument("-verified_zones", "--verified_zones", help="Flag for verified zones", default=True, type=bool)
    parser.add_argument("-verify_threshold", "--verify_threshold", help="Threshold for verification", default=100, type=int)
    parser.add_argument("-remove_zones", "--remove_zones", help="Remove zones flag", default=False, action='store_true')
    parser.add_argument("-category_zones", "--category_zones", help="Flag for category zones", default=False, action='store_true')
    parser.add_argument("-clean_for_training", "--clean_for_training", help="Flag to clean data for training", default=False, action='store_true')
    parser.add_argument("-classes", "--classes", help="List of class IDs", default=[], nargs='+', type=int)
    parser.add_argument("-threshold_file_path", "--threshold_file_path", help="Specify the path to optimal calibrated classes thresholds, specific to each evaluated model ", required=False)
    parser.add_argument("-add_bd_topo", "--add_bd_topo", help="Add topo buildings to annotations", default=False, nargs='+', type=int)
    parser.add_argument("-bd_topo_file", "--bd_topo_file", help="bd topo simplified file", default=None, required=False, type=str)
    
    parser.add_argument("-aigle_output", "--aigle_output", help="Aigle output folder path", default=None, required=False)
    parser.add_argument("-export_gpkg", "--export_gpkg", help="Evaluation option to export a gpkg file with all detections", default=None, required=False)
    parser.add_argument("-export_sql", "--export_sql", help="Export detections to Aigle db", default=None, required=False)
    parser.add_argument("-start_raster_index", "--start_from_raster_index", help="Start from a specific raster block index", default="(0,0)", required=False, type=str)
    parser.add_argument("-start_raster_index_df_results_path", "--start_raster_index_df_results_path", help="Start index associated intermediary images results", default=None, required=False, type=str)
    parser.add_argument("-start_raster_index_df_infos_path", "--start_raster_index_df_infos_path", help="Start index associated intermediary images infos", default=None, required=False, type=str)
    
    parser.add_argument("-s3_bucket_name", "--s3_bucket_name", help="S3 storage bucket name", default=None, required=False, type=str)
    parser.add_argument("-s3_aerial_archive_source_folder", "--s3_aerial_archive_source_folder", help="S3 archive images folder", default=None, required=False, type=str)
    parser.add_argument("-s3_db_topo_archive_source_file", "--s3_db_topo_archive_source_file", help="S3 archive topo file", default=None, required=False, type=str)
    parser.add_argument("-s3_run_folder_path", "--s3_run_folder_path", help="S3 folder to store logs and results", default=None, required=False, type=str)
    
    parser.add_argument("-conv_dataset_input_folder", "--conv_dataset_input_folder", help="Export detections to Aigle db", default=None, required=False)
    parser.add_argument("-conv_dataset_input_type", "--conv_dataset_input_type", help="Export detections to Aigle db", default=None, required=False)
    parser.add_argument("-conv_dataset_input_annotation_file", "--conv_dataset_input_annotation_file", help="Used for inputs datasets of type coco", default=None, required=False)   
    parser.add_argument("-conv_dataset_input_classes_file", "--conv_dataset_input_classes_file", help="Used for inputs datasets of type yolo", default=None, required=False) 
    parser.add_argument("-conv_dataset_output_folder", "--conv_dataset_output_folder", help="Define dataset output path and name", default=None, required=False)
    parser.add_argument("-conv_dataset_output_type", "--conv_dataset_output_type", help="Define dataset output format", default=None, required=False)
    parser.add_argument("-conv_dataset_output_annotation_file", "--conv_dataset_output_annotation_file", help="NOT USED, usage for output datasets of type coco", default=None, required=False)   
    parser.add_argument("-conv_dataset_output_classes_file", "--conv_dataset_output_classes_file", help="NOT USED, usage for output datasets of type yolo", default=None, required=False) 
    parser.add_argument("-conv_image_overlap", "--conv_image_overlap", help="overlab between sub images extracted", default=None, required=False) 
    parser.add_argument("-conv_images_coords_path", "--conv_images_coords", help="NOT USED, usage image geo coordinates file path, it will be converted in new format and save with new dataset", default=None, required=False)
    
    #try :
    args = parser.parse_args()

    # Load configuration from JSON if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Combine args and config with the correct priority
    combined_args = combine_args_with_priority(args, config)

    # Process selection
    process = combined_args.process
    if process == "run_fast_best_segmentations":
        run_fast_aigle_segmentation(combined_args) 
    else:
        print("Unknown process specified.")
        sys.exit(1)
        
    #except Exception as e:
    #    print(f"Error: {e}", file=sys.stderr)
    #    sys.exit(1)
 
if __name__ == "__main__":
    main()