import logging
import random
import time
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from dotenv import load_dotenv
import os
from tqdm import tqdm
from flair_zonal_detection.inference import *
from utils.utils import generate_timestamp
from utils.logs import configure_logging, update_progress
from utils.s3 import *
from utils.export import Exporter
from utils.map import Mapper

logger = logging.getLogger(__name__)

def run_fast_aigle_segmentation(run_config_args) -> None:
    """
    Run inference segmentation using the specified model type on the provided image folder.
    """
    
    logger.info("Initializing process...")        
    load_dotenv()
    data_folder = os.getenv('DATA_FOLDER')
    run_folder = os.getenv('RUN_FOLDER')
    
    debug_mode = run_config_args.debug_mode
    images_type = run_config_args.images_type
    tile_size_px = run_config_args.tile_size_px
    geozone_code = run_config_args.geozones_codes if 'geozones_codes' in run_config_args else 'all'
    input_crs = run_config_args.input_crs
    target_crs = run_config_args.target_crs
    export_sql = run_config_args.export_sql

    dataset_type = run_config_args.dataset_type
    images_folders = run_config_args.images_folders.split(',') if run_config_args.images_folders else []
    db_sources_folder = os.path.abspath(run_config_args.db_sources)
    s3_bucket_name = run_config_args.s3_bucket_name
    s3_aerial_archive_source_folder = run_config_args.s3_aerial_archive_source_folder
    s3_db_topo_archive_source_file = run_config_args.s3_db_topo_archive_source_file
    s3_run_folder_path = run_config_args.s3_run_folder_path 
    model_id = run_config_args.model_id
    version = run_config_args.testset_name + '_' + run_config_args.version
    image_set_name = f"aigle_{images_type}_{dataset_type}_{version}"
    
    experiment_data_folder = os.path.join(data_folder, image_set_name)
    experiment_run_folder = os.path.join(run_folder, image_set_name)
    
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    if not os.path.isdir(run_folder):
        os.makedirs(run_folder)
    if not os.path.isdir(db_sources_folder):
        os.makedirs(db_sources_folder)
    if run_config_args.run_progression_filepath and run_config_args.run_id:
        progression_file_path = run_config_args.run_progression_filepath
    else :
        progression_file_path = None
    
    log_folder, result_folder = prepare_run_folder(experiment_run_folder, progression_file_path)
    images_folders, _, _, _ = prepare_local_data_folder(s3_bucket_name, s3_aerial_archive_source_folder, s3_db_topo_archive_source_file, experiment_data_folder, False, False)
    work_folder = os.path.join(experiment_data_folder, 'work')
    update_progress(25, 'initializing')
    model_ckpt_path, model_threshold_filepath = prepare_local_model_folder(run_folder,model_id)
    update_progress(50, 'initializing')
    
    logger.info("Starting segmentation process...")
    start_total = time.time()
    model_config_path = "configs/config_model_zonal_segmentation.yaml"
    # load aigle segmentation config: config is based on flair config + has geozone info + export infos
    model_config_args = prep_config(model_config_path, model_ckpt_path, model_threshold_filepath, result_folder, log_folder)
    
    # load geozone contour and convert it to same crs as input images
    geozone_geometry_contour = load_geozone_contour(run_config_args)
    
    patch_sizes = compute_patch_sizes(model_config_args)

    start_model = time.time()

    model = build_inference_model(model_config_args, patch_sizes).to(model_config_args['device'])
    logger.info(f"[✓] Loaded model and checkpoint in {time.time() - start_model:.2f}s")
    global_results = []
    # for each raster file build patchs :
    for i, source_image_path in tqdm(enumerate(glob.glob(images_folders+'/*.jp2'))):
        
        raster_results_filepath = os.path.join(result_folder, source_image_path.rsplit('/',1)[-1].replace(".jp2",".gpkg"))
        
        # check if already run
        if os.path.exists(raster_results_filepath):
            logging.warning(f"intermediate result found : {raster_results_filepath} - raster skipped : {source_image_path.rsplit('/',1)[-1]}")
            continue
        
        # run flair zonal inference
        start_slice = time.time()
        
        model_config_args['modalities'][model_config_args['reference_modality']]['input_img_path'] = source_image_path
        model_config_args = initialize_geometry_and_resolutions(model_config_args)
        
        tiles_gdf = generate_patches_from_reference(model_config_args, source_image_path, geozone_geometry_contour)
        logger.info(f"[✓] {source_image_path} Sliced into {len(tiles_gdf)} tiles in {time.time() - start_slice:.2f}s")
        
        if len(tiles_gdf)>0:

            dataset = prep_dataset(model_config_args, tiles_gdf, patch_sizes)
            dataloader = DataLoader(dataset, batch_size=model_config_args['batch_size'], num_workers=model_config_args['num_worker'])

            ref_img = rasterio.open(model_config_args['modalities'][model_config_args['reference_modality']]['input_img_path'])
            
            output_files, temp_paths = init_outputs(model_config_args, ref_img, i)

            start_infer = time.time()
              
            inference_and_write(model, dataloader, tiles_gdf, model_config_args, output_files, ref_img)
            
            gdf_results = raster_to_polygons(output_files,n_jobs=4)
            
            if len(gdf_results) >0 :

                gdf_results.to_file(raster_results_filepath, driver="GPKG")
                global_results.append(raster_results_filepath)
                
            logger.info(f"[✓] Inference completed in {time.time() - start_infer:.2f}s")

    logger.info(f"\n[✓] Total time: {time.time() - start_total:.2f}s")
    logger.info(f"\n[✓] Inference complete. Rasters written to: {work_folder}\n")

    # aggregate all inference
    gdf_results_list = [gpd.read_file(os.path.join(result_folder, file)) for file in os.listdir(result_folder) if file.endswith('.gpkg')]
    global_results_gdf = pd.concat(gdf_results_list, ignore_index=True)
    
    def postprocess_results(global_results_gdf, target_crs, geozone_geometry_contours):
        """
            postprocessing rules to :
            - filter on geozone contour
            - filter classes, 
            - filter object size and 
            - simplify topos
            - convert to target crs
        """
        #  filter on contour, if partially included, then reduce the geometry to inside of the zone
        contour_union = unary_union(geozone_geometry_contours)
        
        # Keep only geometries that intersect the contour
        global_results_gdf = global_results_gdf[global_results_gdf.geometry.intersects(contour_union)]

        # Clip geometries to the contour (only the part inside remains)
        global_results_gdf.loc[:, "geometry"] = global_results_gdf.geometry.intersection(contour_union)

        # filter classes
        clean_results_gdf = global_results_gdf[global_results_gdf.class_id==6]
              
        # simplify geoms .simplify(simplification, preserve_topology=True)
        clean_results_gdf.loc[:,'geometry'] = clean_results_gdf['geometry'].apply(lambda x : x.simplify(tolerance = 1, preserve_topology=True))
        
        # if area is < 50m² remove
        clean_results_gdf = clean_results_gdf[clean_results_gdf.geometry.area > 20]
        
        # TODO improv : calculate the avg confidence of each segmented shape
        clean_results_gdf['confidence'] = [random.uniform(0, 1) for x in range(len(clean_results_gdf))]
        
        clean_results_gdf = clean_results_gdf.to_crs(target_crs)
        
        return clean_results_gdf
    # postprocess results
    clean_results_gdf = postprocess_results(global_results_gdf, run_config_args.target_crs, geozone_geometry_contour)
        
    # Set up exporter and mapper
    description = 'debug_mode' if debug_mode else image_set_name
    export_context = {
        'batch_name': image_set_name,
        'model_id': model_id,
        'export_sql': export_sql,
        'description': description,
        'add_bd_topo': False,
    }

    mapper = Mapper(model_config_args['tasks'][0]['class_names'], export_context['batch_name'])
    exporter = Exporter(input_crs)
    
    # Export results
    exporter.export_to_aigle(clean_results_gdf, target_crs, result_folder, mapper, export_context)
    logger.info("Prediction process complete.")
    update_progress(100, 'exporting')
    s3_runs_path = 's3://'+ s3_bucket_name +'/' + s3_run_folder_path
    upload_run_traces_to_s3(s3_runs_path,experiment_run_folder,image_set_name)
        


